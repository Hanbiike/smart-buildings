"""Occupancy-gated control on the gen-3.1 chain.

The reward charges comfort only for occupied zones and energy/carbon for every
step. Occupancy is zero all night, all weekend and on federal holidays -- over
60% of the episode -- yet the gen-3.1 chain holds the eco band through all of
it (and prestarts + conditions empty buildings on holidays until the static
occupant gate trips two hours after the arrival hour).

This module adds three structural mechanisms, all driven by live observations
and metadata only (no scenario identities, no weather files):

  OccGatedPolicy
    A. Full shutdown whenever the building is empty and no comfort period is
       in force or approaching: AHUs off, water loop cold and still. Zero
       energy, zero comfort risk.
    B. Adaptive prestart: the controller watches how fast the building
       recovers under heat/cool and starts exactly early enough to be in band
       at arrival, instead of a fixed 1-2 h window.
    C. Recoverable overnight floor: in weather too cold to recover from a
       full drift, the night hold is the warmest temperature the observed
       recovery rate cannot climb back from -- not the eco band and not a
       blind shutdown. Mild weather still shuts down fully.
    D. Holiday/early-departure shutdown: occupancy rises monotonically from
       06:00 to 13:00 (arrivals only) and nobody can remain past 19:05, so a
       building still empty an hour into comfort hours is empty all day, and
       a building emptied after 13:00 stays empty. Both shut down immediately.

  OccSatPolicy (subclass, needs full-episode validation before use)
    E. Deep-cold saturation restraint: when full heat is applied for hours
       and the mean zone temperature does not rise, the heat plant is
       saturated and extra airflow only feeds the 30%-outside-air ventilation
       load. Dampers are then capped so the boiler heats zones, not exhaust.
"""

import numpy as np

from smart_buildings.baselines.zone_pi import (
    DsatGen31Policy,
    _outside_kelvin,
)

STEPS_PER_HOUR = 12


class OccGatedPolicy(DsatGen31Policy):
    """Gen-3.1 chain with occupancy-gated shutdown + adaptive prestart."""

    KEY = "occ_gated"

    # Recovery-rate tracking.
    RATE_INIT_K_PER_STEP = 0.08    # ~1 K/h; adapted from the first morning
    RATE_EMA_ALPHA = 0.2           # weight of each new observation
    RATE_MIN = 0.0005
    RATE_MAX = 0.3
    # Prestart sizing.
    PRESTART_SAFETY = 1.3          # start this much earlier than predicted
    PRESTART_MARGIN_STEPS = 6      # +30 min on top of the prediction
    MAX_PRESTART_STEPS = 48        # never prestart more than 4 h early
    FLOOR_CAPABILITY = 0.7         # floor uses 70% of predicted capability
    FLOOR_DEADBAND_K = 0.5
    # Empty-in-comfort-hours detection (holidays / early departures).
    EMPTY_GRACE_STEPS = 12         # 1 h into comfort with nobody -> empty day
    # Occupancy decoding.
    OCC_EPS = 0.5                  # occupied iff obs > running_min + OCC_EPS

    @property
    def name(self):
        return self.KEY

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        names = list(self.field_names)
        self._comfort_now_idx = (names.index("comfort_mode_now")
                                 if "comfort_mode_now" in names else None)
        self._comfort_soon_idx = (names.index("comfort_mode_soon")
                                  if "comfort_mode_soon" in names else None)
        self._occ_zero = None      # running minimum of the occupant signal
        self._rate_heat = self.RATE_INIT_K_PER_STEP
        self._rate_cool = self.RATE_INIT_K_PER_STEP
        self._last_mean = None
        self._comfort_steps = 0    # consecutive comfort_now steps
        self._ever_occupied_today = False
        self._shutdown = False
        self._floor_mode = None    # None | "heat" | "cool"
        self._prestart_active = False
        # Snapshot of every zone temp index for the building mean.
        self._all_zone_obs = [obs_idx for ctrl in self.ahu_controllers
                              for _, obs_idx, _ in ctrl["zones"]]

    # -- observation helpers ------------------------------------------------

    def _occupied(self, obs) -> bool:
        if self._occ_idx is None:
            return True  # no signal: never gate on occupancy
        raw = float(obs[self._occ_idx])
        if self._occ_zero is None:
            # An episode that starts occupied must still read as occupied:
            # raw counts are never negative, so 0.0 is a safe initial floor
            # (a normalized signal corrects it on the first empty night).
            self._occ_zero = min(raw, 0.0)
        elif raw < self._occ_zero:
            self._occ_zero = raw
        return raw > self._occ_zero + self.OCC_EPS

    def _comfort_now(self, obs) -> bool:
        if self._comfort_now_idx is None:
            hour, day = self._get_hour_and_day(obs)
            return (day < 5
                    and self.morning_start_hour <= hour < self.evening_start_hour)
        return float(obs[self._comfort_now_idx]) > 0.5

    def _comfort_soon(self, obs) -> bool:
        if self._comfort_soon_idx is None:
            return False
        return float(obs[self._comfort_soon_idx]) > 0.5

    def _mean_zone_temp(self, obs) -> float:
        if not self._all_zone_obs:
            return float(self.comfort_temp_window[0])
        return float(np.mean([self._denormalize_zone_temp(obs[j])
                              for j in self._all_zone_obs]))

    def _steps_to_comfort(self, obs) -> int | None:
        """Steps until comfort hours, or None if not approaching."""
        if self._comfort_now(obs):
            return 0
        if self._comfort_soon(obs):
            return STEPS_PER_HOUR  # 60-min lookahead by construction
        hour, day = self._get_hour_and_day(obs)
        if day >= 5:
            return None
        if hour < self.morning_start_hour:
            return (self.morning_start_hour - hour) * STEPS_PER_HOUR
        return None

    # -- adaptive rates -----------------------------------------------------

    def _update_rates(self, mean_now: float) -> None:
        if self._last_mean is None:
            self._last_mean = mean_now
            return
        delta = mean_now - self._last_mean
        self._last_mean = mean_now
        if getattr(self, "_max_heat", 0.0) > 0.5 and delta > 0.0002:
            self._rate_heat = float(np.clip(
                (1.0 - self.RATE_EMA_ALPHA) * self._rate_heat
                + self.RATE_EMA_ALPHA * delta, self.RATE_MIN, self.RATE_MAX))
        cool_drive = np.mean(list(getattr(self, "_zone_u", {}).values())) \
            if getattr(self, "_zone_u", {}) else 0.0
        if cool_drive > 0.3 and delta < -0.0002:
            self._rate_cool = float(np.clip(
                (1.0 - self.RATE_EMA_ALPHA) * self._rate_cool
                + self.RATE_EMA_ALPHA * (-delta), self.RATE_MIN, self.RATE_MAX))

    def _needed_prestart(self, mean_now: float) -> int:
        heat_lo, cool_hi = self.comfort_temp_window
        need_heat = max(0.0, (heat_lo + 0.3) - mean_now) / self._rate_heat
        need_cool = max(0.0, mean_now - (cool_hi - 0.3)) / self._rate_cool
        need = max(need_heat, need_cool) * self.PRESTART_SAFETY
        return int(min(self.MAX_PRESTART_STEPS,
                       need + self.PRESTART_MARGIN_STEPS))

    def _floor_window(self) -> tuple[float, float]:
        heat_lo, cool_hi = self.comfort_temp_window
        floor_heat = heat_lo - self.MAX_PRESTART_STEPS * self._rate_heat * self.FLOOR_CAPABILITY
        ceil_cool = cool_hi + self.MAX_PRESTART_STEPS * self._rate_cool * self.FLOOR_CAPABILITY
        return floor_heat, ceil_cool

    # -- mode decision ------------------------------------------------------

    def _decide(self, obs, occupied: bool, mean_now: float) -> str:
        comfort_now = self._comfort_now(obs)
        if comfort_now:
            self._comfort_steps += 1
        else:
            self._comfort_steps = 0
            self._ever_occupied_today = False
        if occupied:
            self._ever_occupied_today = True

        if occupied:
            return "normal"
        if comfort_now:
            # Empty during comfort hours. Occupancy rises monotonically from
            # 06:00 (arrivals only, no departures before 13:00), so a building
            # that was occupied and is now empty stays empty, and a building
            # still empty an hour into comfort hours is a holiday, not a late
            # day -- both shut down.
            if self._ever_occupied_today:
                return "shutdown"
            if self._comfort_steps <= self.EMPTY_GRACE_STEPS:
                return "normal"  # arrivals pending
            return "shutdown"

        steps_left = self._steps_to_comfort(obs)
        if steps_left is not None and steps_left <= self._needed_prestart(mean_now):
            return "prestart"
        floor_heat, ceil_cool = self._floor_window()
        if mean_now < floor_heat - self.FLOOR_DEADBAND_K:
            return "floor_heat"
        if mean_now > ceil_cool + self.FLOOR_DEADBAND_K:
            return "floor_cool"
        return "shutdown"

    # -- PI-hours override ---------------------------------------------------

    def _pi_hours(self, obs):
        if self._prestart_active:
            return True
        if self._floor_mode is not None:
            return False
        # Same timing + occupant gate as the parent, but the gate uses the
        # decoded occupancy signal (raw count or normalized) instead of the
        # parent's raw-only `< 0.5` comparison.
        hour, day = self._get_hour_and_day(obs)
        active = (day < 5
                  and self.morning_start_hour - self.PRESTART_H <= hour
                  < self.evening_start_hour)
        if active and self.OCC_GATE and self._occ_idx is not None:
            if hour >= self.morning_start_hour + 2 and not self._occupied(obs):
                return False
        return active

    # -- actions ---------------------------------------------------------------

    def _shutdown_action(self):
        continuous = np.zeros(self.n_continuous, dtype=np.float32)
        for i in self.temp_indices:
            continuous[i] = -1.0
        for i in self.pressure_indices:
            continuous[i] = -1.0
        for i in self.water_indices:
            continuous[i] = -1.0
        for indices in self.zone_action_indices.values():
            damper_idx = indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = -1.0
            reheat_idx = indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = -1.0
        for i in self.recirculation_indices:
            continuous[i] = 1.0
        return {
            "continuous_action": continuous,
            "discrete_action": np.zeros(self.n_discrete, dtype=np.int32),
        }

    def _enter_shutdown(self):
        self._shutdown = True
        self._on_steps = [0] * len(self.ahu_controllers)
        self._last_on = [None] * len(self.ahu_controllers)
        self._zone_integral = {}
        self._zone_u = {}
        self._max_heat = 0.0
        self._pi_active = False

    def _per_step(self, obs, mean_now: float) -> None:
        """State tracking before the mode decision; subclass hook."""
        self._update_rates(mean_now)

    def get_action(self, obs):
        occupied = self._occupied(obs)
        mean_now = self._mean_zone_temp(obs)
        self._per_step(obs, mean_now)
        mode = self._decide(obs, occupied, mean_now)

        if mode == "shutdown":
            self._prestart_active = False
            self._floor_mode = None
            self._enter_shutdown()
            return self._shutdown_action()

        if mode in ("floor_heat", "floor_cool"):
            self._prestart_active = False
            self._floor_mode = "heat" if mode == "floor_heat" else "cool"
            self._shutdown = False
            floor_heat, ceil_cool = self._floor_window()
            saved_eco = self.eco_temp_window
            saved_ceiling = self.NIGHT_COOL_CEILING
            saved_sensor = self.USE_SENSOR_BAND
            saved_hold = self.HOLD_STEPS
            self.eco_temp_window = (floor_heat, ceil_cool)
            self.NIGHT_COOL_CEILING = False
            # The VAV layer must judge zones against the floor window, not the
            # schedule eco band: otherwise the AHU loop runs for the floor
            # while cold-but-in-eco zones get minimum cold air and no reheat,
            # and the mean never reaches the floor.
            self.USE_SENSOR_BAND = False
            # A cooling-run hold must not extend a floor_cool recovery into a
            # 24-h forced run.
            self.HOLD_STEPS = 1
            try:
                return super().get_action(obs)
            finally:
                self.eco_temp_window = saved_eco
                self.NIGHT_COOL_CEILING = saved_ceiling
                self.USE_SENSOR_BAND = saved_sensor
                self.HOLD_STEPS = saved_hold

        # normal + prestart: the full gen-3.1 chain.
        self._prestart_active = (mode == "prestart")
        self._floor_mode = None
        self._shutdown = False
        return super().get_action(obs)


class OccSatPolicy(OccGatedPolicy):
    """Occupancy gating plus deep-cold saturation restraint.

    EXPERIMENTAL: validate on full episodes before any submission use.

    When full heat has been applied for STALL_STEPS and the mean zone
    temperature has not risen by STALL_MIN_RISE_K, the heat plant is
    saturated: the boiler cannot meet the combined envelope + ventilation
    load however hard the air side works. Extra airflow then only feeds the
    mandatory 30% outside-air stream, so zone dampers are capped at
    SAT_DAMPER_CAP (pressure is already at minimum on the gen-3.1 chain).
    The restraint releases as soon as zones recover or heat demand drops.
    """

    KEY = "occ_sat"

    SAT_ARM_OAT_K = 263.15       # only arm in severe cold (-10 C)
    STALL_STEPS = 24             # 2 h of full heat without recovery
    STALL_MIN_RISE_K = 0.5
    SAT_DAMPER_CAP = 0.5
    SAT_MIN_HEAT = 0.5           # "full heat" threshold on zone demand

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._stall_steps = 0
        self._stall_anchor = None
        self._saturated = False

    def _track_saturation(self, obs, mean_now: float) -> None:
        armed = False
        if self._oat_any is not None:
            try:
                armed = (_outside_kelvin(self, float(obs[self._oat_any]))
                         < self.SAT_ARM_OAT_K)
            except (TypeError, ValueError):
                armed = False
        heat_on = getattr(self, "_max_heat", 0.0) > self.SAT_MIN_HEAT
        if armed and heat_on and not self._shutdown and self._floor_mode is None:
            if self._stall_anchor is None:
                self._stall_anchor = mean_now
                self._stall_steps = 0
            self._stall_steps += 1
            if mean_now - self._stall_anchor >= self.STALL_MIN_RISE_K:
                self._stall_anchor = mean_now
                self._stall_steps = 0
                self._saturated = False
            elif self._stall_steps >= self.STALL_STEPS:
                self._saturated = True
        else:
            self._stall_anchor = None
            self._stall_steps = 0
            self._saturated = False

    def _command_zones(self, continuous, zones, obs, low, high):
        super()._command_zones(continuous, zones, obs, low, high)
        if not self._saturated:
            return
        for vav_id, _, action_indices in zones:
            if not action_indices:
                continue
            if self._zone_u.get(vav_id, 0.0) >= 0:
                continue
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is None:
                continue
            damper = (float(continuous[damper_idx]) + 1.0) * 0.5
            if damper > self.SAT_DAMPER_CAP:
                continuous[damper_idx] = np.float32(2.0 * self.SAT_DAMPER_CAP - 1.0)

    def _per_step(self, obs, mean_now: float) -> None:
        super()._per_step(obs, mean_now)
        self._track_saturation(obs, mean_now)


class OccSat03Policy(OccSatPolicy):
    """Saturation restraint with a tighter damper cap (0.3)."""

    KEY = "occ_sat03"
    SAT_DAMPER_CAP = 0.3


class OccSat07Policy(OccSatPolicy):
    """Saturation restraint with a looser damper cap (0.7)."""

    KEY = "occ_sat07"
    SAT_DAMPER_CAP = 0.7
