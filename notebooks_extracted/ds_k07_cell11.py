"""Main controller: the first baseline plus a cooling-run hold for air handlers.

The organizer AHU loop stops a unit as soon as its average-zone error returns
inside the comfort band, then restarts it at the next excursion. In logs each
AHU switched 50-107 times per day. Stopping a unit that was cooling turned out
to be the baseline's main loss in hot climates. This controller keeps a unit
that was started for cooling running at its last commands for at least
HOLD_STEPS steps. Heating-season runs are unchanged.

Same behaviour as `min_on_cool_<HOLD_STEPS>` in solution/research_policy.py.
Uses only metadata and live observations; no building or weather identities.
"""
import numpy as np


class CoolingHoldPolicy(BaselinePolicy):
    HOLD_STEPS = 288
    # Outside work hours use (eco low, comfort high): no overnight drift up to
    # the eco ceiling in hot climates, full heating setback kept in cold ones.
    # Idea from the public "Neurips 2026 Smart Buildings Smartpolicy" notebook.
    NIGHT_COOL_CEILING = True

    @property
    def name(self):
        return f"cooling_hold_{self.HOLD_STEPS}"

    def _base_action(self, obs):
        if not self.NIGHT_COOL_CEILING:
            return super().get_action(obs)
        hour, day = self._get_hour_and_day(obs)
        if day < 5 and self.morning_start_hour <= hour < self.evening_start_hour:
            return super().get_action(obs)
        saved = self.eco_temp_window
        self.eco_temp_window = (saved[0], self.comfort_temp_window[1])
        try:
            return super().get_action(obs)
        finally:
            self.eco_temp_window = saved

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._on_steps = [0] * len(self.ahu_controllers)
        self._last_on = [None] * len(self.ahu_controllers)

    def get_action(self, obs):
        action = self._base_action(obs)
        cont, disc = action["continuous_action"], action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue
            run = ctrl["run_discrete_idx"]
            # Supply temperature first: its sign tells cooling from heating.
            idx = [j for j in (ctrl["temp_idx"], ctrl["pressure_idx"], ctrl.get("recirc_idx"))
                   if j is not None]
            if (not disc[run] and self._on_steps[i] > 0
                    and self._on_steps[i] < self.HOLD_STEPS and self._last_on[i][0] < 0):
                disc[run] = 1
                cont[idx] = self._last_on[i]
            if disc[run]:
                self._on_steps[i] += 1
                self._last_on[i] = cont[idx].copy()
            else:
                self._on_steps[i] = 0
        return action


class ZonePIPolicy(CoolingHoldPolicy):
    """v5 controller with per-zone PI control to the band centre in work hours.

    A soft proportional zone loop stops the 2-4 K oscillation but leaves zones
    sitting steadily outside the band (no integral action). Here each zone runs
    a PI loop on (temperature - band centre) from one hour before occupancy to
    the end of the work day. One signal u in [-1, 1] drives both actuators:
    the damper opens with |u| (air is needed to cool and to carry reheat) and
    the reheat valve opens with -u. AHUs are kept running in those hours so the
    zones have air to work with. Outside them, v5 logic is unchanged.
    """
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    KP = 0.2
    KI = 0.02
    PRESTART_H = 1
    FORCED_SAT = -0.7       # normalized supply temp for a forced-on AHU (~15 C)
    FORCED_PRESSURE = -0.6  # normalized static pressure for a forced-on AHU
    KEY = "zone_pi"

    @property
    def name(self):
        return self.KEY

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._zone_integral = {}
        self._zone_u = {}
        self._pi_active = False

    def _pi_hours(self, obs):
        hour, day = self._get_hour_and_day(obs)
        return day < 5 and self.morning_start_hour - self.PRESTART_H <= hour < self.evening_start_hour

    def _command_zones(self, continuous, zones, obs, low, high):
        if not self._pi_active:
            for vav_id, _, _ in zones:
                self._zone_integral[vav_id] = 0.0
            return super()._command_zones(continuous, zones, obs, low, high)
        lo_c, hi_c = self.comfort_temp_window
        for vav_id, obs_idx, action_indices in zones:
            if not action_indices:
                continue
            lo, hi = self._zone_band(obs, vav_id, lo_c, hi_c)
            # Before occupancy the sensor band is still the eco band; aim at comfort.
            if hi - lo > (hi_c - lo_c) + 1e-6:
                lo, hi = lo_c, hi_c
            err = self._denormalize_zone_temp(obs[obs_idx]) - self._zone_target(obs, lo, hi)
            integral = self._zone_integral.get(vav_id, 0.0)
            u_raw = self.KP * err + self.KI * integral
            # Conditional integration: stop winding up while saturated.
            if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                integral = float(np.clip(integral + err, -1.0 / self.KI, 1.0 / self.KI))
            self._zone_integral[vav_id] = integral
            u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
            self._max_heat = max(self._max_heat, -u)
            self._zone_u[vav_id] = u
            gain = self.HEAT_DAMPER_GAIN if u < 0 else 1.0
            damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u) * gain
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 2.0 * max(0.0, -u) - 1.0

    MIN_DAMPER = 0.1
    # Damper opening per unit of heating demand (reheated air costs gas).
    HEAT_DAMPER_GAIN = 1.0

    def _zone_target(self, obs, lo, hi):
        """Temperature each zone's PI loop aims at; the band centre by default."""
        return 0.5 * (lo + hi)

    def _base_action(self, obs):
        self._pi_active = self._pi_hours(obs)
        self._max_heat = 0.0
        self._forced = []
        self._zone_u = {}
        action = super()._base_action(obs)
        if self._pi_active:
            cont, disc = action["continuous_action"], action["discrete_action"]
            # Zone reheat needs hot water even when no AHU loop asks for heat.
            if self.water_temp_idx is not None and self._max_heat > 0:
                cont[self.water_temp_idx] = max(cont[self.water_temp_idx],
                                                np.float32(2.0 * self._max_heat - 1.0))
                if self.water_pressure_idx is not None:
                    cont[self.water_pressure_idx] = 1.0
            for i, ctrl in enumerate(self.ahu_controllers):
                if ctrl["zone_obs_indices"] and not disc[ctrl["run_discrete_idx"]]:
                    self._forced.append(i)
                    disc[ctrl["run_discrete_idx"]] = 1
                    cont[ctrl["temp_idx"]] = self.FORCED_SAT
                    if ctrl["pressure_idx"] is not None:
                        cont[ctrl["pressure_idx"]] = self.FORCED_PRESSURE
            if 0 <= self.water_discrete_idx < self.n_discrete:
                disc[self.water_discrete_idx] = 1
        return action

    def get_action(self, obs):
        action = super().get_action(obs)
        # A forced-on unit is not a cooling run; do not let the hold extend it.
        for i in self._forced:
            if self._last_on[i] is not None:
                self._last_on[i][0] = 0.0
        return action


def _outside_kelvin(policy, value):
    """AHU outside-air sensors arrive in raw kelvin; decode only if normalized."""
    value = float(value)
    return value if value > 150.0 else value * policy.outside_std + policy.outside_mean


class ZonePIEconFixPolicy(ZonePIPolicy):
    """pe_lowp with a working economizer (raw-kelvin outside air temperature).

    TransferPolicy decoded the AHU outside-air sensor as if normalized, but it
    arrives in kelvin, so the economizer saw ~3000 K outside and never opened.
    """
    KP = 0.10
    KI = 0.005
    FORCED_PRESSURE = -1.0
    WARM_SAT = None
    WARM_BELOW_K = 283.15
    KEY = "econ_fix"

    def _recirculation(self, obs, ctrl, u, low, high):
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        fixed = obs.copy()
        fixed[idx] = _outside_kelvin(self, obs[idx])
        # Skip TransferPolicy's decoding; call the organizer economizer directly.
        return super(BaselinePolicy, self)._recirculation(fixed, ctrl, u, low, high)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.WARM_SAT is None:
            return action
        cont = action["continuous_action"]
        for i in self._forced:
            ctrl = self.ahu_controllers[i]
            idx = ctrl.get("oat_idx")
            if idx is not None and _outside_kelvin(self, obs[idx]) < self.WARM_BELOW_K:
                cont[ctrl["temp_idx"]] = self.WARM_SAT
        return action


class DemandSatPolicy(ZonePIEconFixPolicy):
    """Supply-air temperature follows the zones' own PI demand in work hours.

    Forced-on AHUs used a fixed ~15 C supply. In cool weather that chills an
    already cool mix, then zone reheat burns gas to warm it back. Here each
    running AHU looks at the mean PI output of its zones (positive = cooling):
    with net heating demand the supply setpoint goes to maximum (no chiller);
    with cooling demand it ramps down to ~15 C, reached at mean demand RAMP
    (RAMP 0 = switch at zero demand).
    """
    RAMP = 0.0
    COLD_SAT = -0.7
    KEY = "dsat_bin"

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._pi_active:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            us = [self._zone_u[v] for v, _, _ in ctrl["zones"] if v in self._zone_u]
            if not us or not disc[ctrl["run_discrete_idx"]]:
                continue
            d = float(np.mean(us))
            frac = (float(np.clip(d / self.RAMP, 0.0, 1.0)) if self.RAMP > 0
                    else (1.0 if d > 0 else 0.0))
            cont[ctrl["temp_idx"]] = np.float32(1.0 + frac * (self.COLD_SAT - 1.0))
        return action


class TunedPIPolicy(ZonePIEconFixPolicy):
    """econ_fix_warmcold plus switchable energy measures, all off by default.

    NIGHT_HEAT_DROP_K: lower the heating setpoint outside PI hours.
    WATER_SCALE: scale the hot-water temperature command (0..1 of range).
    TARGET_SHIFT_K: aim zones this far from the band centre towards the side
        the weather pushes them to (cheaper to hold).
    OCC_GATE: after the arrival hour, run PI hours only while the building
        reports occupants (skips holidays and empty days).
    """
    WARM_SAT = -0.4
    NIGHT_HEAT_DROP_K = 0.0
    WATER_SCALE = 1.0
    TARGET_SHIFT_K = 0.0
    OCC_GATE = False
    KEY = "tuned"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        names = list(self.field_names)
        self._occ_idx = names.index("num_occupants") if "num_occupants" in names else None
        self._oat_any = next((c["oat_idx"] for c in self.ahu_controllers
                              if c.get("oat_idx") is not None), None)

    def _pi_hours(self, obs):
        active = super()._pi_hours(obs)
        if active and self.OCC_GATE and self._occ_idx is not None:
            hour, _ = self._get_hour_and_day(obs)
            if hour >= self.morning_start_hour + 2 and float(obs[self._occ_idx]) < 0.5:
                return False
        return active

    def _zone_target(self, obs, lo, hi):
        centre = 0.5 * (lo + hi)
        if self.TARGET_SHIFT_K <= 0 or self._oat_any is None:
            return centre
        outside = _outside_kelvin(self, obs[self._oat_any])
        shift = min(self.TARGET_SHIFT_K, 0.4 * (hi - lo))
        return centre - shift if outside < centre else centre + shift

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        if self.NIGHT_HEAT_DROP_K and not self._pi_active:
            lo -= self.NIGHT_HEAT_DROP_K
        return lo, hi

    def _ahu_error(self, obs, ctrl, low, high):
        if self.NIGHT_HEAT_DROP_K and not self._pi_active:
            low -= self.NIGHT_HEAT_DROP_K
        return super()._ahu_error(obs, ctrl, low, high)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.WATER_SCALE != 1.0 and self.water_temp_idx is not None:
            cont = action["continuous_action"]
            frac = (float(cont[self.water_temp_idx]) + 1.0) * 0.5 * self.WATER_SCALE
            cont[self.water_temp_idx] = np.float32(2.0 * frac - 1.0)
        return action


class TunedDemandSatPolicy(TunedPIPolicy, DemandSatPolicy):
    """dsat_bin with the TunedPIPolicy switches (all off by default)."""
    WARM_SAT = None  # dsat_bin did not use the cold-weather warm supply
    KEY = "tuned_dsat"
