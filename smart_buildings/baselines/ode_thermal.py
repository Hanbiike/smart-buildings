"""ODE-based (grey-box) temperature control policy. Phase A: L1 + L2.

A self-contained policy that keeps the proven reactive skeleton of
``geometry_aware_baseline`` (signed AHU control, activation gate, zone
damper/reheat layers, outdoor-reset-free binary water loop) and adds a learned
per-zone thermal ODE on top:

    T(t+1) = a·T(t) + b·T_out(t) + g_occ·occupancy(t)
             + g_ex·damper(t)·(T_supply(t) - T(t)) + g_r·u_reheat(t) + c

The parameters are identified ONLINE per zone with recursive least squares
(RLS, forgetting factor), warm-started from geometry priors:

* ``zone_map`` cell counts -> floor area -> thermal capacity -> drift prior
  (small zones drift faster).
* ``floor_plan`` exterior contact fraction -> stronger coupling to outside.

The model is used at two levels:

* L1 (anticipation): every step the policy predicts each zone's temperature
  ``lead_steps`` (30 min) ahead under held commands and drives the zone
  dampers/reheat and the AHU aggregates from the PREDICTED violations, so
  actuation starts before a zone actually leaves the comfort band.
* L2 (morning warm-up): on weekday mornings the controller aims at the
  upcoming comfort window during the ``warmup_minutes`` before
  ``morning_start_hour``, pre-warming zones (and the hot-water loop) so the
  band is reached on time instead of reacting late.

Guardrails: per-zone one-step residuals are tracked with an EMA; a zone whose
model error grows past ``model_mistrust_temp_k`` falls back to the reactive
(margin-only) rule until the residuals recover. Commands are always produced
by the same signed/AHU-idle machinery as before, so an unusable model degrades
to the v2 behavior rather than to nonsense.

Run it with:

    python -m smart_buildings.evaluate --policies ode_thermal \\
        --buildings building_train_1 --weathers atlanta_2

"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

_SPT_ZONE_TEMP = 'zone_air_temperature_sensor'
SP_DAMPER = 'supply_air_damper_percentage_command'
SP_REHEAT = 'reheat_valve_percentage_command'

_STEP_SECONDS = 300.0
_LEAD_STEPS_DEFAULT = 6      # 30 minutes of anticipation


def _trailing_room_number(device_token: str) -> int | None:
    match = re.search(r'(\d+)\s*$', device_token.strip())
    return int(match.group(1)) if match else None


class OdeThermalPolicy:
    """Geometry-prior grey-box thermal model + anticipatory zone control."""

    name = 'ode_thermal'

    # --- Reactive skeleton tunables (same defaults as geometry_zonal v2) ---
    kp: float = 0.4
    ki: float = 0.01
    kd: float = 0.05
    integral_limit: float = 10.0
    u_slew: float = 0.25                 # max |Δu| per step (anti-flip)
    zone_damper_full_k: float = 2.0      # cooling side: K of zone error for a
                                         # fully open damper
    heat_damper_full_k: float = 4.0      # heating side is softer: with 350 K
                                         # water a hard slope bang-bangs zones
                                         # across the whole band in cold climates
    small_zone_threshold: float = 0.6
    small_zone_gain_cap: float = 2.5

    # --- Model / anticipation tunables ------------------------------------
    lead_steps: int = _LEAD_STEPS_DEFAULT
    rls_forgetting: float = 0.999        # ~4 working days of memory
    rls_prior_weight: float = 25.0       # ridge weight on the prior theta
    model_mistrust_temp_k: float = 0.5   # EMA of |1-step residual| that distrusts
    horizon_mistrust_temp_k: float = 1.0  # EMA of |L-step prediction error|
    residual_ema: float = 0.05           # smoothing of the one-step residual
    # Predictive margins replace the reactive ones: the lead time itself is
    # the anticipation, so triggers stay tight around the band.
    predict_margin_k: float = 0.05
    heat_aim_k: float = 0.20          # heating aims this far ABOVE the comfort
                                      # floor: the deadband+margin chain made
                                      # zones hover under the floor all day
                                      # (regret -30..41/workday in Fairbanks).
    heating_activation_k: float = 0.05   # cheap side: comfort-first
    ahu_duty_in_comfort: bool = True     # never starve zones of airflow while
                                         # occupied: idling kills reheat delivery
    cooling_activation_k: float = 0.05   # expensive side: needs a real case
    cool_u_cap: float = 0.75             # cap on how cold the supply gets
    heat_u_cap: float = 0.4              # cap on how hot the supply gets; the
                                         # zone reheat valves do the fine work
    # L2 morning warm-up: the start is adaptive (model-simulated steps to
    # reach the comfort band), capped by this window. A small predicted
    # deficit is left to the cheap reactive layer instead.
    warmup_max_minutes: float = 45.0
    warmup_min_deficit_k: float = 999.0  # warmup disabled: reactive layer is
                                         # fast enough with uncapped valves
    warmup_u_cap: float = 0.5            # cap the supply command during warmup
    reheat_gain: float = 1.0             # K of deficit for full valve: softer
                                         # slope keeps the actuator-limited
                                         # overshoot inside the comfort band
    reheat_deadband_k: float = 0.02      # effectively feathered: the old 0.15
                                         # set the holding point 0.2 K UNDER
                                         # the floor -> chronic below-band time
    warmup_activation_k: float = 0.10    # deficit (K) that wakes the plant

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any,
    ) -> None:
        self.metadata = dict(building_meta_data)
        self.action_spec = action_spec
        self.n_continuous = int(self.metadata['n_continuous_action'])
        self.n_discrete = int(self.metadata['n_discrete_action'])

        self.field_names: list[str] = list(
            self.metadata['observation_field_names']
        )
        self.field_pos = {name: i for i, name in enumerate(self.field_names)}
        self.obs_norm_meta = dict(
            self.metadata.get('observation_normalization_metadata', {}) or {}
        )
        self.action_bounds = dict(
            self.metadata.get('action_normalization_metadata', {}) or {}
        )

        self.zone_action_indices = dict(
            self.metadata.get('zone_action_indices', {}) or {}
        )
        self.ahu_zone_assignments = dict(
            self.metadata.get('ahu_zone_assignments', {}) or {}
        )
        self.ahu_ids = sorted(
            self.ahu_zone_assignments.keys(), key=self._numeric_suffix
        )
        self.n_ahu = len(self.metadata.get('temp_indices', []))
        if self.ahu_ids:
            self.n_ahu = min(self.n_ahu, len(self.ahu_ids))
            self.ahu_ids = self.ahu_ids[: self.n_ahu]

        # --- Time features & schedule windows -----------------------------
        self.hod_sin_idx = self._find_time_index('hod', 'sin')
        self.hod_cos_idx = self._find_time_index('hod', 'cos')
        self.dow_sin_idx = self._find_time_index('dow', 'sin')
        self.dow_cos_idx = self._find_time_index('dow', 'cos')
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

        # --- Observation indices -------------------------------------------
        self.room_temp_obs = self._zone_temp_obs_index()
        zone_meta = self.obs_norm_meta.get(_SPT_ZONE_TEMP)
        if zone_meta:
            self.zone_temp_mean = float(zone_meta.get('sample_mean', 295.0))
            var = float(zone_meta.get('sample_variance', 25.0) or 25.0)
            self.zone_temp_std = math.sqrt(max(var, 1e-9))
        else:
            self.zone_temp_mean, self.zone_temp_std = 295.0, 5.0

        self.ambient_idx = self._first_suffix_index('_outside_air_temperature_sensor')
        self.occupancy_idx = self.field_pos.get('num_occupants')

        # Per-AHU supply-air-temperature observation index.
        # Flat names embed 'ahu_{i}_supply_air_temperature_sensor' somewhere.
        self.ahu_supply_obs: dict[int, int] = {}
        for name, position in self.field_pos.items():
            match = re.search(r'ahu_(\d+)_supply_air_temperature_sensor$', name)
            if match:
                self.ahu_supply_obs.setdefault(int(match.group(1)), position)
        self.ahu_supply_by_position: list[int | None] = []
        for i in range(self.n_ahu):
            self.ahu_supply_by_position.append(
                self.ahu_supply_obs.get(i + 1)
            )

        # Per-AHU supply temp falls back to an ambient reading in Kelvin.
        self.supply_fallback_k = 293.0

        # --- Zone action indices -------------------------------------------
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

        # Room -> AHU position (positional lists in ahu_ids order).
        self.room_to_ahu_pos: dict[int, int] = {}
        for pos, ahu_id in enumerate(self.ahu_ids):
            for room in self.ahu_zone_assignments[ahu_id]:
                room_num = _trailing_room_number(str(room))
                if room_num is not None:
                    self.room_to_ahu_pos[room_num] = pos

        # Room -> damper/reheat OBSERVATION positions (the plant reports what
        # the boxes actually did); used as identification regressors.
        self.room_damper_obs: dict[int, int] = self._suffix_index_map(
            '_supply_air_damper_percentage_command'
        )
        # Bare constants: normalized v -> native = v * 0.5 + 0.5.
        damper_meta = self.obs_norm_meta.get(SP_DAMPER)
        self.damper_norm_mean = float(
            (damper_meta or {}).get('sample_mean', 0.5) or 0.5
        )
        self.damper_norm_std = math.sqrt(
            max(float((damper_meta or {}).get('sample_variance', 0.25) or 0.25), 1e-9)
        )

        # --- AHU controllers (positional lists, AHU order) ------------------
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
        pressure_idx = [int(i) for i in self.metadata.get('pressure_indices', [])]
        self.water_pressure_idx = (
            pressure_idx[-1] if len(pressure_idx) > self.n_ahu else None
        )
        self.water_discrete_idx = self.n_discrete - 1 if self.n_discrete else None

        # --- Geometry priors ------------------------------------------------
        self.room_areas: dict[int, float] = self._compute_zone_areas()
        if not self.room_areas:
            self.room_areas = dict.fromkeys(self.room_temp_obs, 1.0)
        areas = np.asarray(list(self.room_areas.values()), dtype=np.float64)
        self.median_area = float(np.median(areas)) if areas.size else 1.0

        self.room_area_gain: dict[int, float] = {}
        for room, area in self.room_areas.items():
            if area > 0:
                self.room_area_gain[room] = float(
                    np.clip(
                        math.sqrt(self.median_area / area),
                        1.0,
                        self.small_zone_gain_cap,
                    )
                )
            else:
                self.room_area_gain[room] = 1.0
        self.room_small_margin: dict[int, float] = {
            room: (
                0.5 * self.predict_margin_k
                if area < self.small_zone_threshold * self.median_area
                else self.predict_margin_k
            )
            for room, area in self.room_areas.items()
        }

        self.exterior_fraction = self._compute_exterior_fraction()

        # --- Model state: RLS per zone --------------------------------------
        self.theta: dict[int, np.ndarray] = {}
        self.pmat: dict[int, np.ndarray] = {}
        for room in self.room_temp_obs:
            self._init_zone_model(room)
        self.residual: dict[int, float] = dict.fromkeys(self.room_temp_obs, 0.0)
        # Signed one-step bias (actual - prediction), EMA-smoothed; subtracted
        # from horizon predictions to cancel systematic model drift.
        self.bias: dict[int, float] = dict.fromkeys(self.room_temp_obs, 0.0)
        # Horizon (L-step) prediction errors: (deadline_step, predicted K).
        self.pending_predictions: dict[int, list[tuple[int, float]]] = {
            room: [] for room in self.room_temp_obs
        }
        self.horizon_residual: dict[int, float] = dict.fromkeys(
            self.room_temp_obs, 0.0
        )

        # Snapshot of the previous step used as identification regressors.
        self.prev: dict[str, Any] | None = None
        self.step_count = 0

        # --- Per-AHU PID state ----------------------------------------------
        self.pid_state: dict[int, dict[str, float]] = {
            i: {
                'integral': 0.0,
                'prev_error': 0.0,
                'side': 0.0,
                'running': False,
            }
            for i in range(self.n_ahu)
        }

    # ------------------------------------------------------------------ #
    # Parsing helpers (same conventions as geometry_aware_baseline)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numeric_suffix(name: str) -> int:
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0

    def _find_time_index(self, label: str, part: str) -> int | None:
        target = f'{label}_{part}_000'
        if target in self.field_pos:
            return self.field_pos[target]
        suffix = f'{label}_{part}_'
        for name, position in self.field_pos.items():
            if name.startswith(suffix):
                return position
        return None

    @staticmethod
    def _as_pair(value: Any) -> tuple[float, float]:
        try:
            low, high = value
            return float(low), float(high)
        except (TypeError, ValueError):
            return (289.0, 297.0)

    def _first_suffix_index(self, suffix: str) -> int | None:
        for name, position in self.field_pos.items():
            if name.endswith(suffix):
                return position
        return None

    def _suffix_index_map(self, suffix: str) -> dict[int, int]:
        result: dict[int, int] = {}
        for name, position in self.field_pos.items():
            if name.endswith(suffix):
                room_num = _trailing_room_number(name[: -len(suffix)])
                if room_num is not None:
                    result[room_num] = position
        return result

    def _zone_temp_obs_index(self) -> dict[int, int]:
        result: dict[int, int] = {}
        for name, position in self.field_pos.items():
            if name.endswith('_zone_air_temperature_sensor'):
                room_num = _trailing_room_number(
                    name[: -len('_zone_air_temperature_sensor')]
                )
                if room_num is not None:
                    result[room_num] = position
        return result

    def _compute_zone_areas(self) -> dict[int, float]:
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

    def _compute_exterior_fraction(self) -> dict[int, float]:
        """Fraction of each zone's cells that touch the outside (floor_plan)."""
        zone_map = self.metadata.get('zone_map')
        floor_plan = self.metadata.get('floor_plan')
        if zone_map is None or floor_plan is None:
            return {}
        zm = np.asarray(zone_map)
        fp = np.asarray(floor_plan)
        if zm.shape != fp.shape or zm.size == 0:
            return {}
        exterior = (fp == 2).astype(np.uint8)
        # 4-neighborhood dilation: cells adjacent to '2' in floor_plan.
        expanded = np.zeros_like(exterior)
        expanded[1:, :] |= exterior[:-1, :]
        expanded[:-1, :] |= exterior[1:, :]
        expanded[:, 1:] |= exterior[:, :-1]
        expanded[:, :-1] |= exterior[:, 1:]
        expanded |= exterior
        fractions: dict[int, float] = {}
        for room in self.room_temp_obs:
            cells = (zm == room)
            total = int(cells.sum())
            if total > 0:
                fractions[room] = float(cells[expanded.astype(bool)].sum()) / total
            else:
                fractions[room] = 0.3
        return fractions

    # ------------------------------------------------------------------ #
    # Grey-box model: init, RLS, prediction
    # ------------------------------------------------------------------ #

    def _init_zone_model(self, room: int) -> None:
        """Warm-start one zone's ODE parameters from geometry priors."""
        area = self.room_areas.get(room, 1.0)
        # Larger rooms have larger thermal mass -> slower drift. Real学习中
        # RLS refines this within hours; the prior only has to be sane.
        # Passive drift in this simulator is slow: zones barely move without
        # actuation. Priors should assume long time constants (30-120 min);
        # the RLS refines them within hours once actions are observed.
        # Exterior zones lose heat somewhat faster.
        tau_seconds = 2000.0 + 60.0 * math.sqrt(max(area, 1.0))
        tau_seconds = float(np.clip(tau_seconds, 1800.0, 7600.0))
        ext = self.exterior_fraction.get(room, 0.3)
        tau_seconds /= 1.0 + 0.3 * ext
        a0 = math.exp(-_STEP_SECONDS / tau_seconds)
        b0 = 1.0 - a0
        theta = np.array([a0, b0, 0.02, 0.03, 0.01, 0.0], dtype=np.float64)
        self.theta[room] = theta
        p0 = np.eye(len(theta), dtype=np.float64) * 10.0
        self.pmat[room] = p0

    @staticmethod
    def _rls_update(
        theta: np.ndarray, pmat: np.ndarray, x: np.ndarray, y: float,
        forgetting: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """One scalar-output RLS step with forgetting.

        Returns ``(y_hat, error, theta_new, pmat_new)``.
        """
        px = pmat @ x
        denom = forgetting + x @ px
        k_gain = px / max(denom, 1e-9)
        y_hat = float(x @ theta)
        error = y - y_hat
        theta_new = theta + k_gain * error
        pmat_new = (pmat - np.outer(k_gain, px)) / forgetting
        return y_hat, error, theta_new, pmat_new

    def _zone_temperature(self, obs: np.ndarray, obs_idx: int) -> float:
        return float(obs[obs_idx]) * self.zone_temp_std + self.zone_temp_mean

    def _read_ambient(self, obs: np.ndarray) -> float:
        if self.ambient_idx is None:
            return 293.0
        return float(obs[self.ambient_idx])  # AHU fields travel raw (Kelvin)

    def _promotion_snapshot(
        self, obs: np.ndarray, heating_commands: Mapping[int, float],
    ) -> dict[str, Any]:
        """Build the regressor snapshot AFTER acting (used next step)."""
        ambient = self._read_ambient(obs)
        occupancy = (
            float(obs[self.occupancy_idx])
            if self.occupancy_idx is not None and self.occupancy_idx < obs.size
            else 0.0
        )
        temps: dict[int, float] = {}
        dampers: dict[int, float] = {}
        supplies: dict[int, float] = {}
        reheats: dict[int, float] = {}
        for room, obs_idx in self.room_temp_obs.items():
            temps[room] = self._zone_temperature(obs, obs_idx)
            damper_obs = self.room_damper_obs.get(room)
            if damper_obs is not None and damper_obs < obs.size:
                damper_native = (
                    float(obs[damper_obs]) * self.damper_norm_std
                    + self.damper_norm_mean
                )
            else:
                damper_native = 0.3
            dampers[room] = float(np.clip(damper_native, 0.05, 1.0))
            ahu_pos = self.room_to_ahu_pos.get(room)
            supply_idx = (
                self.ahu_supply_by_position[ahu_pos] if ahu_pos is not None else None
            )
            if supply_idx is not None and supply_idx < obs.size:
                supplies[room] = float(obs[supply_idx])  # raw Kelvin
            else:
                supplies[room] = self.supply_fallback_k
            reheats[room] = float(heating_commands.get(room, 0.0))
        return {
            'temps': temps,
            'dampers': dampers,
            'supplies': supplies,
            'reheats': reheats,
            'ambient': ambient,
            'occupancy': occupancy,
        }

    def _run_identification(self, obs: np.ndarray) -> None:
        """RLS update for every zone using the previous snapshot."""
        if self.prev is None:
            return
        prev = self.prev
        forget = self.rls_forgetting
        for room, theta in self.theta.items():
            if room not in prev['temps']:
                continue
            t_prev = prev['temps'][room]
            exch = prev['dampers'][room] * (prev['supplies'][room] - t_prev)
            x = np.array(
                [
                    t_prev,
                    prev['ambient'],
                    prev['occupancy'],
                    exch,
                    prev['reheats'][room],
                    1.0,
                ],
                dtype=np.float64,
            )
            t_now = self._zone_temperature(obs, self.room_temp_obs[room])
            y_hat, error, theta_new, pmat_new = self._rls_update(
                theta, self.pmat[room], x, t_now, forget
            )
            self.theta[room] = theta_new
            self.pmat[room] = pmat_new
            self.residual[room] = (
                (1.0 - self.residual_ema) * self.residual.get(room, 0.0)
                + self.residual_ema * abs(t_now - y_hat)
            )
            # Signed model bias: positive means the model runs colder than
            # the true plant; subtracted over the horizon when predicting.
            decay = 1.0 - self.residual_ema
            self.bias[room] = (
                self.bias.get(room, 0.0) * decay
                + self.residual_ema * (t_now - y_hat)
            )

        # Settle horizon predictions that came due.
        step = self.step_count
        for room, entries in self.pending_predictions.items():
            if not entries:
                continue
            t_now_room = self._zone_temperature(obs, self.room_temp_obs[room])
            due = [p for deadline, p in entries if deadline <= step]
            self.pending_predictions[room] = [
                (deadline, p) for deadline, p in entries if deadline > step
            ]
            if due:
                worst = max(abs(t_now_room - p) for p in due)
                self.horizon_residual[room] = (
                    (1.0 - self.residual_ema) * self.horizon_residual.get(room, 0.0)
                    + self.residual_ema * worst
                )

    def _predict_horizon(
        self,
        room: int,
        start_temp: float,
        ambient: float,
        occupancy: float,
        damper: float,
        supply_temp: float,
        reheat_command: float,
    ) -> float:
        """Iterate the identified one-step model ``lead_steps`` ahead."""
        theta = self.theta.get(room)
        if theta is None:
            return start_temp
        t = float(start_temp)
        exch = damper * (supply_temp - t)
        a, b_out, g_occ, g_ex, g_r, c = [float(v) for v in theta]
        for _ in range(max(1, int(self.lead_steps))):
            t = (
                a * t
                + b_out * ambient
                + g_occ * occupancy
                + g_ex * exch
                + g_r * reheat_command
                + c
            )
            exch = damper * (supply_temp - t)
        return float(t)

    def _steps_to_reach(
        self,
        room: int,
        start_temp: float,
        ambient: float,
        occupancy: float,
        damper: float,
        supply_temp: float,
        target_low: float,
        max_steps: int,
    ) -> int:
        """Model-simulated steps of FULL reheat needed to reach ``target_low``.

        Used to start the morning warm-up exactly on time: earlier would
        waste gas, later would miss the comfort band.
        """
        theta = self.theta.get(room)
        if theta is None:
            return max_steps
        bias = self.bias.get(room, 0.0)
        t = float(start_temp)
        a, b_out, g_occ, g_ex, g_r, c = [float(v) for v in theta]
        for step in range(1, max_steps + 1):
            exch = damper * (supply_temp - t)
            t = (
                a * t
                + b_out * ambient
                + g_occ * occupancy
                + g_ex * exch
                + g_r * 1.0  # full reheat
                + c
            )
            t -= bias  # per-step bias correction
            if t >= target_low:
                return step
        return max_steps

    def _model_trusted(self, room: int) -> bool:
        theta = self.theta.get(room)
        if theta is None:
            return False
        # Physical plausibility of the identified parameters: dynamics from a
        # garbage fit can pass the residual gate (stuck zone -> tiny residual)
        # and then produce nonsense forecasts.
        a, b_out, g_occ, g_ex, g_r, c = [float(v) for v in theta]
        plausible = (
            0.6 <= a <= 1.0
            and 0.0 <= b_out <= 0.4
            and -0.5 <= g_occ <= 1.0
            and -0.5 <= g_ex <= 0.5
            and 0.0 <= g_r <= 0.6
            and abs(c) <= 2.0
        )
        if not plausible:
            return False
        return (
            self.residual.get(room, 0.0) < self.model_mistrust_temp_k
            and self.horizon_residual.get(room, 0.0) < self.horizon_mistrust_temp_k
        )

    # ------------------------------------------------------------------ #
    # Schedule helpers
    # ------------------------------------------------------------------ #

    def _cyclic_feature_to_index(
        self, sin_value: float, cos_value: float, period: int
    ) -> int:
        angle = float(np.arctan2(sin_value, cos_value))
        if angle < 0:
            angle += 2.0 * math.pi
        return int(period * angle / (2.0 * math.pi)) % period

    def _hour_and_day(self, obs: np.ndarray) -> tuple[float, int]:
        hour = 12.0
        day = 2
        if self.hod_sin_idx is not None and self.hod_cos_idx is not None:
            angle = float(
                np.arctan2(obs[self.hod_sin_idx], obs[self.hod_cos_idx])
            )
            if angle < 0:
                angle += 2.0 * math.pi
            hour = 24.0 * angle / (2.0 * math.pi)
        if self.dow_sin_idx is not None and self.dow_cos_idx is not None:
            day = self._cyclic_feature_to_index(
                obs[self.dow_sin_idx], obs[self.dow_cos_idx], 7
            )
        return hour, day

    def _violation(self, temperature: float, low: float, high: float) -> float:
        if temperature < low:
            return low - temperature
        if temperature > high:
            return temperature - high
        return 0.0

    # ------------------------------------------------------------------ #
    # Action mapping
    # ------------------------------------------------------------------ #

    def _norm_bounds(self, setpoint_name: str) -> tuple[float, float] | None:
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
        bounds = self._norm_bounds(setpoint_name)
        if bounds is None:
            if 0.0 <= native_value <= 1.0:
                return 2.0 * float(native_value) - 1.0
            return 0.0
        lo, hi = bounds
        if hi <= lo:
            return 0.0
        return float(np.clip((native_value - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0))

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32).ravel()

        # 1) Identify first: consumes the previous snapshot with fresh facts.
        self._run_identification(obs)

        hour, day = self._hour_and_day(obs)

        ambient = self._read_ambient(obs)
        occupancy = (
            float(obs[self.occupancy_idx])
            if self.occupancy_idx is not None and self.occupancy_idx < obs.size
            else 0.0
        )

        # Per-zone supply temperature (from the zone's AHU observation).
        supply_by_room: dict[int, float] = {}
        current_damper: dict[int, float] = {}
        for room, obs_idx in self.room_temp_obs.items():
            ahu_pos = self.room_to_ahu_pos.get(room)
            supply_idx = (
                self.ahu_supply_by_position[ahu_pos] if ahu_pos is not None else None
            )
            supply_by_room[room] = (
                float(obs[supply_idx])
                if supply_idx is not None and supply_idx < obs.size
                else self.supply_fallback_k
            )
            damper_obs = self.room_damper_obs.get(room)
            if damper_obs is not None and damper_obs < obs.size:
                current_damper[room] = float(
                    np.clip(
                        float(obs[damper_obs]) * self.damper_norm_std
                        + self.damper_norm_mean,
                        0.05,
                        1.0,
                    )
                )
            else:
                current_damper[room] = 0.3

        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        discrete = np.zeros(self.n_discrete, dtype=np.int32)

        any_active = False
        any_heating = False
        any_warmup = False
        heating_commands: dict[int, float] = {}

        is_workday = day < 5
        in_comfort_window = (
            is_workday
            and self.morning_start_hour <= hour < self.evening_start_hour
        )
        steps_to_start = (
            (self.morning_start_hour - hour) * 12.0
            if is_workday and hour < self.morning_start_hour
            else float('inf')
        )
        warmup_max_steps = self.warmup_max_minutes / _STEP_SECONDS * 60.0

        # 2) Zone layer driven by PREDICTED violations (L1) + adaptive
        #    morning warm-up (L2, per AHU).
        for ctrl in self.ahu_controllers:
            zones = [
                (room, obs_idx)
                for room, obs_idx in
                ((r, self.room_temp_obs.get(r, None)) for r in ctrl['zones'])
                if obs_idx is not None and obs_idx < obs.size
            ]
            if not zones:
                continue

            # L2: how many full-reheat steps does the slowest zone need to
            # reach the comfort band by morning start?
            warmup_needed = 0
            if is_workday and hour < self.morning_start_hour:
                for room, obs_idx in zones:
                    if self._model_trusted(room):
                        k = self._steps_to_reach(
                            room,
                            self._zone_temperature(obs, obs_idx),
                            ambient,
                            occupancy,
                            current_damper[room],
                            supply_by_room[room],
                            self.comfort_temp_window[0],
                            int(warmup_max_steps),
                        )
                    else:
                        k = int(warmup_max_steps)
                    # Skip the warmup for shallow deficits: reaching the band
                    # a few minutes late costs almost no regret, while the
                    # boiler gas is real.
                    deficit = self.comfort_temp_window[0] - self._zone_temperature(
                        obs, obs_idx
                    )
                    if deficit < self.warmup_min_deficit_k:
                        k = 0
                    warmup_needed = max(warmup_needed, k)
            warmup_active = (
                is_workday
                and hour < self.morning_start_hour
                and steps_to_start
                <= max(float(warmup_needed) + 1.0, 2.0)
            )

            if warmup_active:
                target_low, target_high = self.comfort_temp_window
                # Heating aims slightly above the floor so the deadband/margin
                # chain cannot park zones just under it (that point is still
                # outside the band and pays regret every step in cold
                # climates; overshoot up to the ceiling is free comfort-wise).
                aim_low = target_low + self.heat_aim_k
            elif in_comfort_window:
                target_low, target_high = self.comfort_temp_window
                aim_low = target_low + self.heat_aim_k
            else:
                target_low, target_high = self.eco_temp_window
                aim_low = target_low

            pred_temps: list[float] = []
            actual_temps: list[float] = []
            any_deficit = False
            for room, obs_idx in zones:
                temperature = self._zone_temperature(obs, obs_idx)
                actual_temps.append(temperature)
                margin = self.room_small_margin.get(room, self.predict_margin_k)
                if self._model_trusted(room):
                    reheat_hold = (
                        self.prev['reheats'].get(room, 0.0)
                        if self.prev is not None
                        else 0.0
                    )
                    predicted = self._predict_horizon(
                        room,
                        temperature,
                        ambient,
                        occupancy,
                        current_damper[room],
                        supply_by_room[room],
                        reheat_hold,  # coasting: hold the commanded reheat
                    )
                    # Cancel systematic model drift over the horizon.
                    # bias > 0: the model runs colder than the plant (actual
                    # one-step error positive) -> raise the forecast.
                    predicted += self.bias.get(room, 0.0) * float(
                        self.lead_steps
                    )
                    # Early on, the parameters are priors; discount the
                    # forecast's deviation from the current temperature until
                    # the RLS has seen a day of actuation data.
                    ramp = min(1.0, self.step_count / 240.0)
                    if ramp < 1.0:
                        predicted = temperature + (
                            predicted - temperature
                        ) * ramp
                    # A 30-minute forecast cannot be wilder than a generous
                    # per-step drift; clip so one bad fit cannot flip the
                    # whole air handler.
                    max_drift = 0.5 * float(self.lead_steps)
                    predicted = float(
                        np.clip(
                            predicted,
                            temperature - max_drift,
                            temperature + max_drift,
                        )
                    )
                    # Register for horizon-error tracking; the actual
                    # temperature is compared when the deadline arrives.
                    self.pending_predictions[room].append(
                        (self.step_count + int(self.lead_steps), predicted)
                    )
                else:
                    predicted = temperature
                pred_temps.append(predicted)
                cold_error = max(0.0, (aim_low - margin) - predicted)
                hot_error = max(0.0, predicted - (target_high - margin))
                if warmup_active and cold_error > 0.0:
                    any_deficit = True

                self._command_zone_actions(
                    continuous,
                    room,
                    cold_error,
                    hot_error,
                )
                heating_commands[room] = float(
                    min(1.0, max(0.0, cold_error - self.reheat_deadband_k))
                    if cold_error > self.reheat_deadband_k
                    else 0.0
                )

            # AHU aggregate: heating reacts to the GROUP'S mean predicted
            # temperature (pid-style) so a few cold zones never drag the
            # plant into heating the entire airstream; cooling keeps the
            # mean-of-per-zone-violations trigger because early cooling is
            # what wins the afternoon (and it is now protected from model
            # garbage by the plausibility/clip/slew guards).
            mean_pred = float(np.mean(pred_temps))
            aggregate_cold = max(0.0, target_low - mean_pred)
            mean_hot_violation = float(
                np.mean([max(0.0, pt - target_high) for pt in pred_temps])
            )
            aggregate_hot = mean_hot_violation
            if warmup_active and any_deficit:
                aggregate_cold = max(aggregate_cold, self.warmup_activation_k)
            if warmup_active:
                any_warmup = True
            if aggregate_cold >= aggregate_hot:
                aggregate = aggregate_cold
                sign = +1.0
                activation = self.heating_activation_k
            else:
                aggregate = aggregate_hot
                sign = -1.0
                activation = self.cooling_activation_k

            state = self.pid_state[ctrl['run_discrete_idx']]
            if aggregate <= activation:
                # COMFORT-WINDOW DUTY: an idled AHU leaves its zones without
                # airflow, so open reheat valves cannot deliver any heat (the
                # Fairbanks catastrophe: zones sat 1-2 K under the floor with
                # the valve fully open, AHU mean in band -> fan off). Run a
                # neutral supply at minimum fan pressure instead; the zone
                # dampers and reheat valves modulate per zone from there.
                if (
                    in_comfort_window
                    and self.ahu_duty_in_comfort
                ):
                    state['running'] = True
                    state['u_prev'] = 0.0
                    continuous[ctrl['temp_idx']] = 0.0
                    if ctrl['pressure_idx'] is not None:
                        continuous[ctrl['pressure_idx']] = -1.0
                    discrete[ctrl['run_discrete_idx']] = 1
                    any_active = True
                    continue
                # Idle the handler; also stops a previously-running one,
                # preventing the running flag from sticking on forever.
                if state.get('running'):
                    state['running'] = False
                state['integral'] *= 0.5
                state['prev_error'] = 0.0
                state['side'] = 0.0
                # Re-activation starts from neutral, not from the idle
                # 'cold' supply setpoint, otherwise the slew limiter would
                # keep the supply cold for several steps while heating.
                state['u_prev'] = 0.0
                continuous[ctrl['temp_idx']] = -1.0
                if ctrl['pressure_idx'] is not None:
                    continuous[ctrl['pressure_idx']] = -1.0
                continue

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

            u_raw = (
                self.kp * aggregate
                + self.ki * state['integral']
                + self.kd * derivative
            )
            u_mag = float(min(abs(u_raw), 1.0))
            if sign < 0.0:
                # Cooling is expensive and, pushed too hard, starts fighting
                # the reheat layer; cap how cold the supply setpoint goes.
                u_mag = min(u_mag, self.cool_u_cap)
            else:
                # Heating the whole supply stream is the single most
                # expensive action in the building; zone reheat valves
                # should carry the load, the AHU only avoids overcooling.
                u_mag = min(u_mag, self.heat_u_cap)
                if warmup_active and hour < self.morning_start_hour:
                    u_mag = min(u_mag, self.warmup_u_cap)
            u = sign * u_mag
            # Slew limit: no ±1 flips in a single step. A model that switches
            # sides should ramp the plant, not slam it.
            u_prev = float(state.get('u_prev', u))
            u = float(np.clip(u, u_prev - self.u_slew, u_prev + self.u_slew))
            state['u_prev'] = u
            state['running'] = aggregate > 0.0

            continuous[ctrl['temp_idx']] = u
            if ctrl['pressure_idx'] is not None:
                # Fan effort follows the ACTUAL violation, the forecast only
                # leads the setpoint. Anticipating early with a deeper supply
                # while the measured violation is small otherwise pays for
                # full-pressure fans (dominant energy cost) for little regret.
                if sign > 0.0:
                    actual_agg = max(
                        0.0, target_low - float(np.mean(actual_temps))
                    )
                else:
                    actual_agg = float(
                        np.mean([max(0.0, t - target_high) for t in actual_temps])
                    )
                pressure = float(
                    np.clip(
                        min(
                            2.0 * actual_agg - 1.0,
                            2.0 * abs(u) - 1.0,
                        ),
                        -1.0,
                        1.0,
                    )
                )
                continuous[ctrl['pressure_idx']] = pressure
            discrete[ctrl['run_discrete_idx']] = 1
            any_active = True
            if sign > 0.0:
                any_heating = True

        # 3) Plant-level overrides.
        for idx in self.recirc_indices:
            continuous[idx] = 1.0

        if any_warmup:
            # Pre-warm the water loop so the reheat layer has capacity when
            # the comfort window opens.
            any_heating = True

        # Zone valves need HOT water even when no AHU loop asked for heat
        # (duty-run handlers with one cold zone inside an in-band mean).
        if heating_commands and max(heating_commands.values()) > 0.05:
            any_heating = True

        if self.water_temp_idx is not None:
            continuous[self.water_temp_idx] = 1.0 if any_heating else -1.0
        if self.water_pressure_idx is not None:
            continuous[self.water_pressure_idx] = 1.0 if any_active else -1.0
        if self.water_discrete_idx is not None:
            discrete[self.water_discrete_idx] = 1 if any_active else 0

        # 4) Refresh the identification snapshot with what we just did.
        self.prev = self._promotion_snapshot(obs, heating_commands)
        self.step_count += 1

        return {
            'continuous_action': np.clip(continuous, -1.0, 1.0).astype(np.float32),
            'discrete_action': discrete,
        }

    # ------------------------------------------------------------------ #
    # Zone layer
    # ------------------------------------------------------------------ #

    def _command_zone_actions(
        self,
        continuous: np.ndarray,
        room: int,
        cold_error: float,
        hot_error: float,
    ) -> float:
        """Drive one zone's damper + reheat valve; returns the reheat level."""
        gain = self.room_area_gain.get(room, 1.0)
        # Separate slopes: the chilling delivery wants the faster damper, the
        # heating side needs room to feather air that is usually near-ambient.
        damper_native = min(
            1.0,
            cold_error / self.heat_damper_full_k * gain
            + hot_error / self.zone_damper_full_k * gain,
        )
        damper_idx = self.room_damper_action_idx.get(room)
        if damper_idx is not None:
            continuous[damper_idx] = self._to_norm(SP_DAMPER, damper_native)

        # Proportional reheat with a deadband: shallow deficits keep the
        # valve shut (gas is not worth a few hundredths of K), deeper ones
        # open proportionally, deep ones fully.
        if cold_error <= self.reheat_deadband_k:
            reheat_native = 0.0
        else:
            effective = cold_error - self.reheat_deadband_k
            reheat_native = float(min(1.0, effective * self.reheat_gain))
        reheat_idx = self.room_reheat_action_idx.get(room)
        if reheat_idx is not None:
            continuous[reheat_idx] = self._to_norm(SP_REHEAT, reheat_native)
        return reheat_native
