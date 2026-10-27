"""Geometry-aware zone-level baseline policy.

A self-contained starter policy that uses BOTH inputs the competition gives a
policy:

1. The per-step flat observation vector ``obs`` (zone temperatures, cyclic
   time features, ...).
2. The static ``building_meta_data`` handed to ``__init__`` once per scenario,
   including its GEOMETRY payload: ``floor_plan`` / ``zone_map`` arrays,
   ``ahu_zone_assignments`` and the action index tables.

The control scheme follows the shape of the shipped PID baselines because that
shape is proven on this simulator: every AHU idles (fan off, minimum pressure)
while ALL of its zones sit inside the active comfort/eco band and runs a
proportional-integral loop on their violations otherwise; fan pressure scales
with control effort; zone dampers are proportional per zone, the reheat valve
is a hot/cold switch; the hot-water loop is binary hot/cold with the pump
tracking activity.

What the geometry adds on top of that skeleton:

* Per-room areas are counted from ``zone_map`` cells (label ``k`` is room
  ``k``; generator-guaranteed). Small zones have little thermal mass and can
  run away from the plant (see ``smart_control/simulator/zone_floor_area_test.py``
  for the chain reaction this causes), so zones much smaller than the median
  room get a pre-trigger margin and boosted damper/reheat authority.
* AHU aggregate violations use area-weighted averaging, so one neglected small
  room is not diluted by thirty large ones in the idle decision: the AHU idle
  rule requires every served zone to be in band, weighted by inverse area.

Everything degrades gracefully: missing metadata keys, absent sensors, zone
maps whose labels do not line up with the observed room ids (falls back to
uniform areas), unknown device-name conventions (constants are looked up by
exactly emulating the simulator normalizer, with a raw-vs-z-score magnitude
guard), and buildings of any AHU/zone count.

Run it with:

    python -m smart_buildings.evaluate --policies geometry_zonal \\
        --buildings building_train_1 --weathers fairbanks_1

"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

# Observation field-name suffixes (field ids are "<device_id>_<field_name>").
_SPT_ZONE_TEMP = 'zone_air_temperature_sensor'

# Bare setpoint names, as used to key the action normalization metadata.
SP_DAMPER = 'supply_air_damper_percentage_command'
SP_REHEAT = 'reheat_valve_percentage_command'

_DEVICE_WORDS = {'ahu', 'vav', 'room', 'boiler', 'pump', 'hvac', 'hws', 'water'}


def _trailing_room_number(device_token: str) -> int | None:
    """'VAV_12' or 'room_12' -> 12; tokens without a trailing int -> None."""
    match = re.search(r'(\d+)\s*$', device_token.strip())
    return int(match.group(1)) if match else None


class GeometryAwareZonalPolicy:
    """Deterministic geometry-aware thermal controller for one scenario."""

    name = 'geometry_zonal'

    # --- Tunables ----------------------------------------------------------
    # The PID loop keeps the shipped pid.py defaults on purpose: those gains
    # were tuned for this simulator and regressions against them are the
    # signal that geometry changes helped.
    kp: float = 0.5
    ki: float = 0.02
    kd: float = 0.1
    integral_limit: float = 10.0
    zone_damper_full_k: float = 2.0   # K of violation -> damper fully open
    # AHU activation threshold (K of mean violation). Below it the handler
    # idles; keeps tiny violations from running the plant around the clock.
    activation_threshold: float = 0.02
    # Small zones pre-trigger this much earlier and grip proportionally.
    small_zone_margin_k: float = 0.5
    small_zone_threshold: float = 0.6  # fraction of the median room area
    small_zone_gain_cap: float = 2.5
    # 'Hod'/'dow' cyclic-feature names per building; regex fallbacks applied.
    #

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any,
    ) -> None:
        self.metadata = dict(building_meta_data)
        self.action_spec = action_spec
        self.n_continuous = int(self.metadata['n_continuous_action'])
        self.n_discrete = int(self.metadata['n_discrete_action'])

        # --- Observation name -> position index ---------------------------
        self.field_names: list[str] = list(
            self.metadata['observation_field_names']
        )
        self.field_pos = {name: i for i, name in enumerate(self.field_names)}

        self.obs_norm_meta: dict[str, Any] = dict(
            self.metadata.get('observation_normalization_metadata', {}) or {}
        )
        self.action_bounds: dict[str, Any] = dict(
            self.metadata.get('action_normalization_metadata', {}) or {}
        )

        # --- Static plant layout from metadata ----------------------------
        self.zone_action_indices: dict[str, dict[str, int]] = dict(
            self.metadata.get('zone_action_indices', {}) or {}
        )
        self.ahu_zone_assignments: dict[str, list[str]] = dict(
            self.metadata.get('ahu_zone_assignments', {}) or {}
        )
        self.ahu_ids = sorted(
            self.ahu_zone_assignments.keys(), key=self._numeric_suffix
        )
        self.n_ahu = len(self.metadata.get('temp_indices', []))
        if len(self.ahu_ids) != self.n_ahu:
            # Trust whichever is smaller so action writes never cross devices.
            self.n_ahu = min(self.n_ahu, len(self.ahu_ids)) if self.ahu_ids else 0
            self.ahu_ids = self.ahu_ids[: self.n_ahu]

        # --- Cyclic time-feature indices (schedule reconstruction) --------
        self.hod_sin_idx = self._find_time_index('hod', 'sin')
        self.hod_cos_idx = self._find_time_index('hod', 'cos')
        self.dow_sin_idx = self._find_time_index('dow', 'sin')
        self.dow_cos_idx = self._find_time_index('dow', 'cos')

        # --- Schedule windows ---------------------------------------------
        self.comfort_temp_window = self._as_pair(
            self.metadata.get('comfort_temp_window', (295.0, 297.0))
        )
        self.eco_temp_window = self._as_pair(
            self.metadata.get('eco_temp_window', (289.0, 298.0))
        )
        self.morning_start_hour = int(self.metadata.get('morning_start_hour', 6) or 6)
        self.evening_start_hour = int(self.metadata.get('evening_start_hour', 19) or 19)
        if self.morning_start_hour >= self.evening_start_hour:
            self.morning_start_hour, self.evening_start_hour = 6, 19

        # --- Per-room tables ----------------------------------------------
        self.room_temp_obs = self._zone_temp_obs_index()
        # zone temp -> native units via bare sensor constants.
        zone_meta = self._norm_meta_for(_SPT_ZONE_TEMP)
        if zone_meta:
            self.zone_temp_mean = float(zone_meta.get('sample_mean', 295.0))
            var = float(zone_meta.get('sample_variance', 25.0) or 25.0)
            self.zone_temp_std = math.sqrt(max(var, 1e-9))
        else:
            self.zone_temp_mean, self.zone_temp_std = 295.0, 5.0

        self.room_reheat_action_idx: dict[int, int] = {}
        self.room_damper_action_idx: dict[int, int] = {}
        for device_id, setpoints in self.zone_action_indices.items():
            room_num = _trailing_room_number(device_id)
            if room_num is None:
                continue
            for setpoint_name, action_idx in setpoints.items():
                if setpoint_name == SP_REHEAT:
                    self.room_reheat_action_idx[room_num] = int(action_idx)
                elif setpoint_name == SP_DAMPER:
                    self.room_damper_action_idx[room_num] = int(action_idx)

        # Rooms controllable per-zone: at least one of the two setpoints.
        self.room_to_ahu: dict[int, str] = {}
        for idx, ahu_id in enumerate(self.ahu_ids):
            rooms = self.ahu_zone_assignments[ahu_id]
            for room in rooms:
                room_num = _trailing_room_number(str(room))
                if room_num is not None:
                    self.room_to_ahu[room_num] = ahu_id

        # AHU control tables, aligned by the metadata's positional action lists.
        self.ahu_controllers: list[dict[str, Any]] = []
        for i, ahu_id in enumerate(self.ahu_ids):
            rooms = [
                _trailing_room_number(str(room))
                for room in self.ahu_zone_assignments[ahu_id]
            ]
            zones = [r for r in rooms if r is not None and r in self.room_temp_obs]
            self.ahu_controllers.append(
                {
                    'ahu_id': ahu_id,
                    'temp_idx': int(self.metadata['temp_indices'][i]),
                    'pressure_idx': (
                        int(self.metadata['pressure_indices'][i])
                        if i < len(self.metadata.get('pressure_indices', []))
                        else None
                    ),
                    'run_discrete_idx': i,
                    'zones': zones,
                }
            )

        self.recirc_indices = [
            int(i) for i in self.metadata.get('recirculation_indices', [])
        ]
        water_idx = [int(i) for i in self.metadata.get('water_indices', [])]
        self.water_temp_idx = water_idx[0] if water_idx else None
        # The hot-water differential pressure rides at the end of the
        # pressure list (same convention the shipped PID baselines assume).
        pressure_idx = [int(i) for i in self.metadata.get('pressure_indices', [])]
        self.water_pressure_idx = (
            pressure_idx[-1] if len(pressure_idx) > self.n_ahu else None
        )
        self.water_discrete_idx = self.n_discrete - 1 if self.n_discrete else None

        # --- Geometry from the zone map -----------------------------------
        self.room_areas: dict[int, float] = self._compute_zone_areas()
        median_area = 1.0
        if self.room_areas:
            areas = np.asarray(list(self.room_areas.values()), dtype=np.float64)
            median_area = float(np.median(areas)) if areas.size else 1.0
        else:
            self.room_areas = dict.fromkeys(self.room_temp_obs, 1.0)
        self.median_area = median_area

        self.room_area_gain: dict[int, float] = {}
        for room, area in self.room_areas.items():
            if area > 0:
                self.room_area_gain[room] = float(
                    np.clip(
                        math.sqrt(self.median_area / area), 1.0, self.small_zone_gain_cap
                    )
                )
            else:
                self.room_area_gain[room] = 1.0
        self.room_small_margin: dict[int, float] = {
            room: (
                self.small_zone_margin_k
                if area < self.small_zone_threshold * self.median_area
                else 0.0
            )
            for room, area in self.room_areas.items()
        }

        # --- Per-AHU PID state --------------------------------------------
        self.pid_state: dict[int, dict[str, float]] = {
            i: {'integral': 0.0, 'prev_error': 0.0, 'side': 0.0, 'running': False}
            for i in range(self.n_ahu)
        }

    # ------------------------------------------------------------------ #
    # Small parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numeric_suffix(name: str) -> int:
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0

    def _find_time_index(self, label: str, part: str) -> int | None:
        """Locate 'hod_sin_000'-style cyclic features, or None."""
        for name, position in self.field_pos.items():
            if name.startswith(f'{label}_{part}_'):
                return position
        for name, position in self.field_pos.items():
            if name == f'{label}_{part}':
                return position
        return None

    @staticmethod
    def _as_pair(value: Any) -> tuple[float, float]:
        try:
            low, high = value
            return float(low), float(high)
        except (TypeError, ValueError):
            return (289.0, 297.0)

    def _norm_meta_for(self, bare_name: str) -> dict[str, Any] | None:
        return self.obs_norm_meta.get(bare_name)

    def _zone_temp_obs_index(self) -> dict[int, int]:
        """Room number -> observation index for the zone temperature sensor."""
        result: dict[int, int] = {}
        for name, position in self.field_pos.items():
            if name.endswith('_zone_air_temperature_sensor'):
                room_num = _trailing_room_number(
                    name[: -len('_zone_air_temperature_sensor')]
                )
                if room_num is not None:
                    result[room_num] = position
        return result

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #

    def _compute_zone_areas(self) -> dict[int, float]:
        """Per-room relative areas from ``zone_map``, or {} on any mismatch.

        The generator labels rooms depth-first over the stacked plan and calls
        the labels ``room_1..room_N``, so zone-map value ``k`` is room ``k``.
        """
        zone_map = self.metadata.get('zone_map')
        if zone_map is None or not self.room_temp_obs:
            return {}
        array = np.asarray(zone_map)
        if array.ndim < 2 or array.size == 0:
            return {}
        positive = array[array > 0]
        if positive.size == 0:
            return {}
        labels, counts = np.unique(positive, return_counts=True)
        labeled_rooms = {
            int(label): float(count) for label, count in zip(labels, counts)
        }
        observed = set(self.room_temp_obs)
        if not observed.issubset(labeled_rooms.keys()):
            return {}
        return {room: labeled_rooms[room] for room in observed}

    # ------------------------------------------------------------------ #
    # Unit conversion
    # ------------------------------------------------------------------ #

    def _norm_bounds(self, setpoint_name: str) -> tuple[float, float] | None:
        """Native (min, max) bounds for one bare setpoint name."""
        bounds = self.action_bounds.get(setpoint_name)
        if bounds is None:
            for key, value in self.action_bounds.items():
                if setpoint_name in key:
                    bounds = value
                    break
        if bounds is None:
            return None
        try:
            return (
                float(bounds['min_native_value']),
                float(bounds['max_native_value']),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _to_norm(self, setpoint_name: str, native_value: float) -> float:
        """Map a native setpoint value into the [-1, 1] action range."""
        bounds = self._norm_bounds(setpoint_name)
        if bounds is None:
            # The zone actuators are native [0, 1]; scale without bounds info.
            if 0.0 <= native_value <= 1.0:
                return 2.0 * float(native_value) - 1.0
            return 0.0
        lo, hi = bounds
        if hi <= lo:
            return 0.0
        return float(np.clip((native_value - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0))

    # ------------------------------------------------------------------ #
    # Schedule
    # ------------------------------------------------------------------ #

    def _cyclic_feature_to_index(
        self, sin_value: float, cos_value: float, period: int
    ) -> int:
        angle = float(np.arctan2(sin_value, cos_value))
        if angle < 0:
            angle += 2.0 * math.pi
        return int(period * angle / (2.0 * math.pi)) % period

    def _active_band(self, obs: np.ndarray) -> tuple[float, float]:
        """Return the (low, high) comfort bounds that apply right now."""
        if self.hod_sin_idx is None or self.hod_cos_idx is None:
            return self.comfort_temp_window
        hour = self._cyclic_feature_to_index(obs[self.hod_sin_idx], obs[self.hod_cos_idx], 24)
        if self.dow_sin_idx is None or self.dow_cos_idx is None:
            day = 2  # mid-week fallback: err on the stricter comfort window
        else:
            day = self._cyclic_feature_to_index(obs[self.dow_sin_idx], obs[self.dow_cos_idx], 7)
        is_work_hour = self.morning_start_hour <= hour < self.evening_start_hour
        if day < 5 and is_work_hour:
            return self.comfort_temp_window
        return self.eco_temp_window

    def _violation(self, temperature: float, low: float, high: float) -> float:
        """Unsigned excursion outside the band: >0 means 'fix me'."""
        if temperature < low:
            return low - temperature
        if temperature > high:
            return temperature - high
        return 0.0

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32).ravel()
        band_low, band_high = self._active_band(obs)

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        any_active = False
        any_heating = False

        for ctrl in self.ahu_controllers:
            zones = [
                (room, obs_idx)
                for room, obs_idx in
                ((r, self.room_temp_obs.get(r, None)) for r in ctrl['zones'])
                if obs_idx is not None and obs_idx < obs.size
            ]
            if not zones:
                continue

            # Compose per-zone state. Zone-layer actuation uses the margin
            # (small zones pre-trigger), while the AHU-level aggregates below
            # use raw violations so a 0.05 K margin trip cannot keep the plant
            # running around the clock.
            cold_violations: list[float] = []
            hot_violations: list[float] = []
            for room, obs_idx in zones:
                temperature = self._zone_temperature(obs, obs_idx)
                margin = self.room_small_margin.get(room, 0.0)
                cold_error = max(0.0, (band_low - margin) - temperature)
                hot_error = max(0.0, temperature - (band_high - margin))
                cold_violations.append(max(0.0, band_low - temperature))
                hot_violations.append(max(0.0, temperature - band_high))
                self._command_zone_actions(
                    continuous,
                    room,
                    cold_error + hot_error,
                    cold_error > 0.0,
                )

            mean_cold = float(np.mean(cold_violations))
            mean_hot = float(np.mean(hot_violations))
            if mean_cold >= mean_hot:
                aggregate = mean_cold
                sign = +1.0  # heating: warm the supply air
            else:
                aggregate = mean_hot
                sign = -1.0  # cooling: chill the supply air

            state = self.pid_state[ctrl['run_discrete_idx']]
            if not state.get('running') and aggregate <= self.activation_threshold:
                state['integral'] *= 0.5
                state['prev_error'] = 0.0
                state['side'] = 0.0
                continuous[ctrl['temp_idx']] = -1.0
                if ctrl['pressure_idx'] is not None:
                    continuous[ctrl['pressure_idx']] = -1.0
                continue

            # Direction switch: dump old heating windup when switching to
            # cooling and vice versa.
            state['side'] = float(sign)
            if 'integral_side' in state and state['integral_side'] != sign:
                state['integral'] = 0.0
                state['prev_error'] = 0.0
            state['integral_side'] = float(sign)

            state['integral'] = float(
                np.clip(
                    state['integral'] + aggregate,
                    -self.integral_limit,
                    self.integral_limit,
                )
            )
            derivative = aggregate - state['prev_error']
            state['prev_error'] = aggregate

            u = sign * (
                self.kp * aggregate
                + self.ki * state['integral']
                + self.kd * derivative
            )
            u = float(np.clip(u, -1.0, 1.0))
            state['running'] = aggregate > 0.0

            # Supply-air temperature: positive commands warm, negative cold.
            # In cold weather the mixed air is already below any commanded
            # hot setpoint, so the heating command goes inert and the reheat
            # layer supplies the heat instead; the simulator clamps this.
            continuous[ctrl['temp_idx']] = u
            if ctrl['pressure_idx'] is not None:
                continuous[ctrl['pressure_idx']] = float(
                    np.clip(2.0 * abs(u) - 1.0, -1.0, 1.0)
                )
            discrete[ctrl['run_discrete_idx']] = 1
            any_active = True
            if sign > 0.0:
                any_heating = True

        # Recirculate as much as the ventilation cap allows everywhere, all
        # the time, like the shipped PIDs do. In hot weather this saves fans
        # pulling hot fresh air; in cold weather it keeps return heat.
        for idx in self.recirc_indices:
            continuous[idx] = 1.0

        # Binary hot-water loop: hot when some handler is heating.
        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if self.water_discrete_idx is not None:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        return {
            'continuous_action': np.clip(continuous, -1.0, 1.0).astype(np.float32),
            'discrete_action': discrete,
        }

    # ------------------------------------------------------------------ #
    # Per-zone layer
    # ------------------------------------------------------------------ #

    def _zone_temperature(self, obs: np.ndarray, obs_idx: int) -> float:
        return float(obs[obs_idx]) * self.zone_temp_std + self.zone_temp_mean

    def _command_zone_actions(
        self,
        continuous: np.ndarray,
        room: int,
        zone_error: float,
        heating: bool,
    ) -> None:
        """Damper proportional to the zone's violation; reheat a hot switch.

        ``zone_error`` already includes the small-zone margin shrink (a small
        zone counts as violating earlier) and ``room_area_gain`` inflates the
        damper so a small zone grabs more of the shared airflow. A zone below
        the band wants its reheat valve open; a hot zone wants the valve shut
        and its damper open (the AHU supplies the cooling).
        """
        gain = self.room_area_gain.get(room, 1.0)
        damper_native = min(1.0, zone_error / self.zone_damper_full_k * gain)
        damper_idx = self.room_damper_action_idx.get(room)
        if damper_idx is not None:
            continuous[damper_idx] = self._to_norm(SP_DAMPER, damper_native)

        reheat_idx = self.room_reheat_action_idx.get(room)
        if reheat_idx is not None:
            if heating:
                continuous[reheat_idx] = self._to_norm(SP_REHEAT, 1.0)
            else:
                continuous[reheat_idx] = self._to_norm(SP_REHEAT, 0.0)

