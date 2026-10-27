"""Per-zone PI/HVAC chain, ported verbatim from a leaderboard-verified
public notebook lineage (public LB 0.01853 with the k10 configuration).

Chain (each class adds one idea; Apache-2.0 organizer ``pid_plus`` underneath):

  BaselinePolicy      Organizer PIDWaterResetPolicy with per-zone comfort
                      bands read from sensors, mean-zone aggregation and the
                      outside-air temperature decoded (raw kelvin) before the
                      economizer decision.
  CoolingHoldPolicy   Keeps a unit that was started *for cooling* running at
                      its last commands for HOLD_STEPS steps (organizer loop
                      chatters 50-107 on/off per day otherwise); optional night
                      cooling ceiling (eco high -> comfort high outside work
                      hours so hot buildings do not drift overnight).
  ZonePIPolicy        Per-zone PI to the band centre in work hours (one signal
                      drives the damper |u| and the reheat valve -u); AHUs are
                      forced on through those hours (FORCED_SAT/FORCED_PRESSURE).
  ZonePIEconFixPolicy Economizer fed the AHU outside-air sensor decoded as
                      kelvin (the organizer transfer policy decoded it as
                      normalized, so the economizer never opened).
  DemandSatPolicy     Supply-air temperature follows the zones' mean PI demand
                      in PI hours (net heating demand -> max supply SAT, no
                      chiller; cooling demand -> ramp to COLD_SAT).
  TunedPIPolicy       Switchable energy measures (night heating drop, water
                      scale, target shift, occupancy gate), all off by default.
  TunedDemandSatPolicy

Uses only metadata and live observations; no scenario identities or weather
files. Zone/AHU counts come from  each building's metadata.
"""

import numpy as np
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


def _outside_kelvin(policy, value):
    """AHU outside-air sensors arrive in raw kelvin; decode only if normalized."""
    value = float(value)
    return value if value > 150.0 else value * policy.outside_std + policy.outside_mean


class BaselinePolicy(PIDWaterResetPolicy):
    """First baseline: sensor comfort bands, mean zone, decoded weather.

    The version submitted as ``transfer`` on the public leaderboard (0.06439).
    """

    USE_SENSOR_BAND = True
    # Worst-zone aggregation increased loss in both full-length pilot scenarios.
    USE_WORST_ZONE = False

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        stats = self.observation_normalization_metadata.get(
            "outside_air_temperature_sensor", {}
        )
        self.outside_mean = float(stats.get("sample_mean", 0.0))
        self.outside_std = float(stats.get("sample_std", 1.0))

    def _recirculation(self, obs, ctrl, u, low, high):
        # The parent's economizer expects kelvin; the observation is standardized.
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        decoded = obs.copy()
        decoded[idx] = obs[idx] * self.outside_std + self.outside_mean
        return super()._recirculation(decoded, ctrl, u, low, high)

    def get_action(self, obs):
        action = super().get_action(obs)
        # Enforce the gateway's dtype, shape and bounds contract.
        for key, dtype in (("continuous_action", np.float32),
                           ("discrete_action", np.int32)):
            space = self.action_spec[key]
            action[key] = np.clip(action[key], space.low, space.high).astype(dtype)
        return action


class CoolingHoldPolicy(BaselinePolicy):
    """Baseline plus a cooling-run hold (and the optional night ceiling)."""

    HOLD_STEPS = 288
    # Outside work hours use (eco low, comfort high): no overnight drift up to
    # the eco ceiling in hot climates, full heating setback kept in cold ones.
    NIGHT_COOL_CEILING = True

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._on_steps = [0] * len(self.ahu_controllers)
        self._last_on = [None] * len(self.ahu_controllers)

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
    """Per-zone PI control to the band centre in work hours.

    One signal u in [-1, 1] drives both actuators: the damper opens with |u|
    (air is needed to cool and to carry reheat) and the reheat valve opens
    with -u. AHUs are kept running in those hours so the zones have air to
    work with. Outside them, the cooling-hold logic is unchanged.

    Class defaults below match the k10 leaderboard probe (public LB 0.01853).
    """

    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    KP = 0.1
    KI = 0.005
    PRESTART_H = 1
    FORCED_SAT = -0.7       # normalized supply temp for a forced-on AHU (~15 C)
    FORCED_PRESSURE = -1.0  # normalized static pressure for a forced-on AHU
    MIN_DAMPER = 0.1
    # Damper opening per unit of heating demand (reheated air costs gas).
    HEAT_DAMPER_GAIN = 1.0
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
        return (day < 5
                and self.morning_start_hour - self.PRESTART_H <= hour
                < self.evening_start_hour)

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
            # Before occupancy the sensor band is still the eco band; aim
            # at comfort.
            if hi - lo > (hi_c - lo_c) + 1e-6:
                lo, hi = lo_c, hi_c
            err = self._denormalize_zone_temp(obs[obs_idx]) - self._zone_target(
                obs, lo, hi
            )
            integral = self._zone_integral.get(vav_id, 0.0)
            u_raw = self.KP * err + self.KI * integral
            # Conditional integration: stop winding up while saturated.
            if abs(u_raw) < 1.0 or np.sign(err) != np.sign(u_raw):
                integral = float(np.clip(integral + err, -1.0 / self.KI,
                                         1.0 / self.KI))
            self._zone_integral[vav_id] = integral
            u = float(np.clip(self.KP * err + self.KI * integral, -1.0, 1.0))
            self._max_heat = max(self._max_heat, -u)
            self._zone_u[vav_id] = u
            damper, reheat = self._zone_output_map(u)
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is not None:
                continuous[damper_idx] = 2.0 * damper - 1.0
            reheat_idx = action_indices.get("reheat_valve_percentage_command")
            if reheat_idx is not None:
                continuous[reheat_idx] = 2.0 * reheat - 1.0

    def _zone_target(self, obs, lo, hi):
        """Temperature each zone's PI loop aims at; the band centre by default."""
        return 0.5 * (lo, hi)[0] + 0.5 * (lo, hi)[1]

    def _zone_output_map(self, u):
        """(damper_01, reheat_01) from PI signal u in [-1, 1].

        Coupled by default: the damper opens with |u| and the reheat valve
        with -u simultaneously. Override for decoupled (dual-maximum) maps.
        """
        gain = self.HEAT_DAMPER_GAIN if u < 0 else 1.0
        damper = self.MIN_DAMPER + (1.0 - self.MIN_DAMPER) * abs(u) * gain
        reheat = max(0.0, -u)
        return damper, reheat

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


class ZonePIEconFixPolicy(ZonePIPolicy):
    """Zone PI with a working economizer (raw-kelvin outside air temperature).

    The organizer transfer policy decoded the AHU outside-air sensor as if
    normalized, but it arrives in kelvin, so the economizer saw ~3000 K
    outside and never opened.
    """

    WARM_SAT = None
    WARM_BELOW_K = 283.15
    KEY = "econ_fix"

    def _recirculation(self, obs, ctrl, u, low, high):
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        fixed = obs.copy()
        fixed[idx] = _outside_kelvin(self, obs[idx])
        # Skip BaselinePolicy's decoding; call the organizer economizer directly.
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
    """Supply-air temperature follows the zones' own PI demand in PI hours.

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
    """Zone PI + economizer plus switchable energy measures, all off by default.

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


class ZonePiDsatPolicy(TunedDemandSatPolicy):
    """The ds-k07 submission configuration (newer iteration, LB unconfirmed).

    KP 0.07 / KI 0.0035, two-hour prestart, night heating drop 2 K, min-fan
    pressure on forced units, demand-driven supply-air temperature.
    """

    KP = 0.07
    KI = 0.0035
    HOLD_STEPS = 288
    NIGHT_COOL_CEILING = True
    FORCED_PRESSURE = -1.0
    NIGHT_HEAT_DROP_K = 2.0
    PRESTART_H = 2
    KEY = "zone_pi_dsat"


class DsatTargetShiftPolicy(ZonePiDsatPolicy):
    """ds-k07 with TARGET_SHIFT_K enabled: aim zones towards the cheap side of
    the band based on outdoor temperature (cold outside -> lower target, hot
    outside -> higher target). Data-driven, no climate identities."""

    TARGET_SHIFT_K = 1.0
    KEY = "dsat_shift10"


class DsatWaterScalePolicy(ZonePiDsatPolicy):
    """ds-k07 with the hot-water command scaled to 70% of range (gas saving)."""

    WATER_SCALE = 0.7
    KEY = "dsat_water07"


class DsatOccGatePolicy(ZonePiDsatPolicy):
    """ds-k07 with occupant-gated PI hours (skip heating empty buildings)."""

    OCC_GATE = True
    KEY = "dsat_occgate"


class DsatShiftDownPolicy(ZonePiDsatPolicy):
    """TARGET_SHIFT only towards the lower band edge: cold outside -> aim
    below centre (heating economy). The upward shift their mechanic applies
    in hot climates is clipped away - tight hot bands lose more comfort
    margin than the chiller saves (singapore_8 regressed +25%)."""

    TARGET_SHIFT_K = 1.0
    KEY = "dsat_shiftdown"

    def _zone_target(self, obs, lo, hi):
        centre = 0.5 * (lo + hi)
        if self.TARGET_SHIFT_K <= 0 or self._oat_any is None:
            return centre
        outside = _outside_kelvin(self, obs[self._oat_any])
        if outside >= centre:
            return centre
        shift = min(self.TARGET_SHIFT_K, 0.4 * (hi - lo))
        return centre - shift


class DsatShiftHalfPolicy(ZonePiDsatPolicy):
    """ds-k07 with a small symmetric TARGET_SHIFT (0.5 K)."""

    TARGET_SHIFT_K = 0.5
    KEY = "dsat_shift05"


class DsatShiftDownOccGatePolicy(DsatShiftDownPolicy):
    """Down-shift heating economy + occupant-gated PI hours combined."""

    OCC_GATE = True
    KEY = "dsat_shiftdown_occgate"


class DsatPressureMinPolicy(ZonePiDsatPolicy):
    """Minimum static pressure (200 Pa) for every RUNNING AHU, not only the
    forced-on ones. Fan power is the single biggest electricity cost in hot
    climates; the organizer baseline runs mid-scale pressure on its cooling
    units."""

    KEY = "dsat_pressmin"

    def _base_action(self, obs):
        action = super()._base_action(obs)
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            if ctrl["pressure_idx"] is None:
                continue
            if disc[ctrl["run_discrete_idx"]]:
                cont[ctrl["pressure_idx"]] = np.float32(-1.0)
        return action


class DsatRampPolicy(ZonePiDsatPolicy):
    """Ramped demand-SAT (0.5) instead of the binary deep-cool switch: supply
    only cools as deep as the zones' mean demand actually requires."""

    RAMP = 0.5
    KEY = "dsat_ramp05"


class DsatPressureMinRampPolicy(DsatPressureMinPolicy):
    """pressmin + ramped demand-SAT combined."""

    RAMP = 0.5
    KEY = "dsat_pressmin_ramp05"


class DsatNightDrop3Policy(ZonePiDsatPolicy):
    """Deeper night heating setback (3 K): gas saved through the eco nights,
    tracked for morning catch-up cost."""

    NIGHT_HEAT_DROP_K = 3.0
    KEY = "dsat_nightdrop3"


class DsatPMShift05Policy(DsatPressureMinPolicy):
    """pressmin + symmetric TARGET_SHIFT 0.5 K: the two mechanisms are
    independent (fan/pressure waste vs reheat demand depth)."""

    TARGET_SHIFT_K = 0.5
    KEY = "dsat_pm_shift05"


class DsatPMShiftDownPolicy(DsatPressureMinPolicy):
    """pressmin + down-only TARGET_SHIFT 1.0 K (up-shift provably costs
    comfort in tight hot bands - singapore_8 +25%)."""

    TARGET_SHIFT_K = 1.0
    KEY = "dsat_pm_shiftdown"

    def _zone_target(self, obs, lo, hi):
        centre = 0.5 * (lo + hi)
        if self.TARGET_SHIFT_K <= 0 or self._oat_any is None:
            return centre
        outside = _outside_kelvin(self, obs[self._oat_any])
        if outside >= centre:
            return centre
        shift = min(self.TARGET_SHIFT_K, 0.4 * (hi - lo))
        return centre - shift


class DsatSatMildPolicy(DsatPMShiftDownPolicy):
    """Gentler cooling SAT (-0.4 ~ 295 K): less chiller per unit air, argued
    for hot climates where the PI already has deep dampers."""

    COLD_SAT = -0.4
    KEY = "dsat_sat_mild"


class DsatSatDeepPolicy(DsatPMShiftDownPolicy):
    """Deepest cooling SAT (-1.0 = 285 K): max cooling per unit air."""

    COLD_SAT = -1.0
    KEY = "dsat_sat_deep"


class DsatRamp035Policy(DsatPMShiftDownPolicy):
    """Demand-SAT glide saturating early (0.35): light demand gets a mild
    supply, sustained heavy demand saturates deep. atlanta wanted mild
    (-23%), singapore wanted deep (-4%): depth follows load, no identities."""

    RAMP = 0.35
    KEY = "dsat_ramp035"


class DsatRamp035DeepPolicy(DsatRamp035Policy):
    """ramp035 saturating at the deepest SAT."""

    COLD_SAT = -1.0
    KEY = "dsat_ramp035_deep"


class DsatAdaptiveSatPolicy(DsatPMShiftDownPolicy):
    """SAT depth follows a slow EMA of the zones' cooling demand, not the
    instant mean: light loads earn a mild supply (atlanta -23%), sustained
    heavy loads saturate deep (singapore -4%), and the slow EMA prevents the
    mid-range cycling that made instant ramps worse (+8..12%). Depth 1 gives
    the deepest SAT, zero leaves +1.0 (neutral cooling)."""

    EMA_KEEP = 0.97          # per-step decay (5-min steps, ~hours scale)
    EMA_SATURATION = 0.35    # sustained demand at which depth reaches 1
    COLD_SAT = -1.0
    KEY = "dsat_adaptive"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._sat_depth: dict[int, float] = {}

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._pi_active:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            us = [self._zone_u[v] for v, _, _ in ctrl["zones"] if v in self._zone_u]
            if not us or not disc[ctrl["run_discrete_idx"]]:
                continue
            d = max(0.0, float(np.mean(us)))
            depth = self._sat_depth.get(i, 0.0)
            depth = self.EMA_KEEP * depth + (1.0 - self.EMA_KEEP) * d
            self._sat_depth[i] = depth
            frac = float(np.clip(depth / self.EMA_SATURATION, 0.0, 1.0))
            cont[ctrl["temp_idx"]] = np.float32(1.0 + frac * (self.COLD_SAT - 1.0))
        return action


class DsatGen1OccGatePolicy(DsatPMShiftDownPolicy):
    """gen-1 (pressmin + down-shift) with occupant-gated PI hours."""

    OCC_GATE = True
    KEY = "dsat_gen1_occ"


class DsatWaterPumpFracPolicy(DsatPMShiftDownPolicy):
    """Hot-water differential pressure follows the reset fraction instead of
    pinning the whole scale whenever any zone asks for heat: pump head is
    spent water-side effort the zones may not need."""

    KEY = "dsat_wpump_frac"

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.water_temp_idx is None or self.water_pressure_idx is None:
            return action
        cont = action["continuous_action"]
        frac = (float(cont[self.water_temp_idx]) + 1.0) * 0.5
        if frac > 0.02:
            cont[self.water_pressure_idx] = np.float32(2.0 * frac - 1.0)
        return action


class DsatWaterPumpIdleOffPolicy(DsatPMShiftDownPolicy):
    """Pump (discrete + head) off entirely when no zone wants heat: the PI
    skeleton currently pins the water pump ON for all PI hours even at zero
    heating demand."""

    KEY = "dsat_wpump_idleoff"

    def _base_action(self, obs):
        action = super()._base_action(obs)
        cont = action["continuous_action"]
        frac = ((float(cont[self.water_temp_idx]) + 1.0) * 0.5
                if self.water_temp_idx is not None else 0.0)
        if frac < 0.05 and self._max_heat < 0.05:
            if self.water_pressure_idx is not None:
                cont[self.water_pressure_idx] = np.float32(-1.0)
            if 0 <= self.water_discrete_idx < self.n_discrete:
                action["discrete_action"][self.water_discrete_idx] = 0
        return action


class DsatGen2Policy(DsatGen1OccGatePolicy):
    """The Stage-2 assembly: pressmin, down-shift, occupant gate. Renamed
    here so follow-up experiments subclass the standing configuration."""

    KEY = "dsat_gen2"


class DsatResetFull25Policy(DsatGen2Policy):
    """Water reset saturates at 2.5 K mean shortfall instead of 4.0: hotter
    water sooner, bet on reheat coil delivering before zones deep-freeze."""

    RESET_FULL_K = 2.5
    KEY = "dsat_reset_full25"


class DsatResetEffort2Policy(DsatGen2Policy):
    """Double weight on the air-side saturation term of the water reset:
    heat the water up when handlers are pinned, even at modest shortfalls."""

    RESET_EFFORT_WEIGHT = 2.0
    KEY = "dsat_reset_effort2"


class DsatResetTightPolicy(DsatGen2Policy):
    """Tighter reset follow (deadband 0.1 K instead of 0.25)."""

    RESET_DEADBAND_K = 0.1
    KEY = "dsat_reset_tight"


class DsatForcedPressureUpPolicy(DsatGen2Policy):
    """Forced-on AHUs at mid-scale static pressure: in the deep-cold escalation
    the 200 Pa minimum airflow cannot deliver reheat against stochastic
    convection, so the building loses its thermal mass day over day."""

    FORCED_PRESSURE = -0.6
    KEY = "dsat_fp_06"


class DsatForcedPressureHighPolicy(DsatGen2Policy):
    """Forced-on AHUs at high static pressure (upper quartile of range)."""

    FORCED_PRESSURE = -0.2
    KEY = "dsat_fp_02"


class DsatShiftDeepPolicy(DsatGen2Policy):
    """Deeper down-shift: FB workday zones ride ~293.4 with zero regret, so
    the 0.4*band cap (1.2 K) leaves gas on the table. Cap raised to 0.6*band
    (1.8 K)."""

    TARGET_SHIFT_K = 2.0
    SHIFT_FRACTION = 0.6
    KEY = "dsat_shift_deep"

    def _zone_target(self, obs, lo, hi):
        centre = 0.5 * (lo + hi)
        if self.TARGET_SHIFT_K <= 0 or self._oat_any is None:
            return centre
        outside = _outside_kelvin(self, obs[self._oat_any])
        if outside >= centre:
            return centre
        shift = min(self.TARGET_SHIFT_K, self.SHIFT_FRACTION * (hi - lo))
        return centre - shift


class DsatShiftDeepestPolicy(DsatShiftDeepPolicy):
    """Cap 0.8*band (2.4 K down-shift)."""

    SHIFT_FRACTION = 0.8
    KEY = "dsat_shift_deepest"


class DsatPrestartGatePolicy(DsatGen2Policy):
    """Extend the occupant gate over the prestart hours: if the building has
    been empty for the last day (24-h occupant mean ~0), skip the prestart
    burn (holidays heat an empty plant for no one). First workday after a
    holiday pays the catch-up from the arrival hour instead."""

    KEY = "dsat_prestart_gate"

    OCC_24H_ALPHA = 1.0 - 1.0 / 288.0   # ~24-h EMA at 5-min steps

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._occ_ema = 0.0

    def _pi_hours(self, obs):
        if self._occ_idx is not None:
            occ = float(obs[self._occ_idx])
            self._occ_ema = (self.OCC_24H_ALPHA * self._occ_ema
                             + (1.0 - self.OCC_24H_ALPHA) * occ)
            if self._occ_ema < 0.5:
                hour, _ = self._get_hour_and_day(obs)
                if hour < self.morning_start_hour + 2:
                    return False
        return super()._pi_hours(obs)


class DsatMixedSatPolicy(DsatGen2Policy):
    """Per-AHU SAT depth from load mixedness: an air handler whose zones
    carry sustained heating demand pays twice for deep cooling SAT (chill the
    mix, then reheat it with gas), so its cooling supply stays mild. Purely
    cooling handlers keep the reference deep endpoint. Heating shares are a
    24-h EMA over each handler's own zones - no climate identities."""

    KEY = "dsat_sat_mixed"

    AHU_EMA_ALPHA = 1.0 - 1.0 / 288.0    # ~24 h
    HEAT_SHARE_GATE = 0.05
    MILD_SAT = -0.4                       # vs reference COLD_SAT -0.7
    COOL_ACTIVE_GATE = 0.02               # d above which cooling counts as on
    COOL_SHARE_DEEP_GATE = None           # None = base never goes deep

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._heat_ema: dict[int, float] = {}
        self._cool_ema: dict[int, float] = {}

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._pi_active:
            return action
        cont = action["continuous_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            us = [self._zone_u[v] for v, _, _ in ctrl["zones"] if v in self._zone_u]
            if not us:
                continue
            heat = float(np.mean([u < 0.0 for u in us]))
            cool_on = 1.0 if float(np.mean(us)) > self.COOL_ACTIVE_GATE else 0.0
            alpha = self.AHU_EMA_ALPHA
            self._heat_ema[i] = alpha * self._heat_ema.get(i, heat) + (1.0 - alpha) * heat
            self._cool_ema[i] = alpha * self._cool_ema.get(i, cool_on) + (1.0 - alpha) * cool_on
            mixed = self._heat_ema[i] > self.HEAT_SHARE_GATE
            sat = cont[ctrl["temp_idx"]]
            if mixed and float(sat) < 0.5:   # cooling demand active now
                cont[ctrl["temp_idx"]] = np.float32(self.MILD_SAT)
            elif (self.COOL_SHARE_DEEP_GATE is not None and not mixed
                  and self._cool_ema[i] > self.COOL_SHARE_DEEP_GATE
                  and float(sat) < 0.5):
                cont[ctrl["temp_idx"]] = np.float32(self.DEEP_SAT)
        return action


class DsatSatMixedDeepPolicy(DsatMixedSatPolicy):
    """Mixed-load handlers stay mild; sustained cooling-dominant handlers
    (SG/PHX heavy: chiller is everything, no reheat to betray) saturate the
    deepest SAT. Light pure-cooling handlers keep the reference depth."""

    COOL_SHARE_DEEP_GATE = 0.7
    DEEP_SAT = -1.0
    KEY = "dsat_sat_mixed_deep"


class DsatGen31Policy(DsatSatMixedDeepPolicy):
    """Standing freeze candidate: gen-2 skeleton + per-AHU SAT depth from
    load mixedness (mild) and cooling dominance (deep)."""

    KEY = "dsat_gen31"

    def _dump_state(self):
        """Per-step internal state, replayable from the observation sequence.

        Every channel is a deterministic function of the obs history this
        policy itself observed, so a shadow instance run in deployment (or a
        replay of a dumped obs sequence) reproduces it exactly: per-zone PI
        signal and integral, per-AHU mixedness/cooling EMAs and the hold
        counter, plus global PI-hours and occupant EMA.
        """
        zones: list[float] = []
        ahus: list[float] = []
        for i, ctrl in enumerate(self.ahu_controllers):
            hold = getattr(self, "_on_steps", None)
            for vav_id, _, _ in ctrl["zones"]:
                zones.append(float(self._zone_u.get(vav_id, 0.0)))
                zones.append(float(self._zone_integral.get(vav_id, 0.0)))
            on = hold[i] if hold is not None and i < len(hold) else 0
            ahus.append(float(self._heat_ema.get(i, 0.0)))
            ahus.append(float(self._cool_ema.get(i, 0.0)))
            ahus.append(float(on) / float(self.HOLD_STEPS))
        zones.extend(ahus)
        zones.append(1.0 if self._pi_active else 0.0)
        zones.append(float(getattr(self, "_occ_ema", 0.0)))
        return np.asarray(zones, dtype=np.float32)


class DsatPrestart1Policy(DsatGen2Policy):
    """Shorter prestart (1 h): less off-hours burn, morning catch-up risk."""

    PRESTART_H = 1
    KEY = "dsat_prestart1"


class DsatPrestart3Policy(DsatGen2Policy):
    """Longer prestart (3 h): head start for the coldest catch-ups."""

    PRESTART_H = 3
    KEY = "dsat_prestart3"


class DsatGen3Policy(DsatMixedSatPolicy):
    """Stage-3 assembly standing on gen-2: pressmin + down-shift + occupant
    gate + mixedness-driven per-AHU SAT depth. Freeze candidate for the
    global validation and the submission checkpoint."""

    KEY = "dsat_gen3"


# ---------------------------------------------------------------------------
# gen-4 screen: single-knob re-parents of the gen-2-era experiments onto the
# gen-3.1 chain (mixedness-SAT + pressmin + down-shift skeleton). Each class
# is gen-3.1 plus exactly one change, so a screen on the 30x864 table
# attributes the delta to the knob, not to a bundle.
# ---------------------------------------------------------------------------


class DsatG4ResetFull25Policy(DsatGen31Policy):
    """Water reset saturates at 2.5 K mean shortfall instead of 4.0: hotter
    water sooner — bet on the reheat coil delivering before zones freeze."""

    RESET_FULL_K = 2.5
    KEY = "g4_reset_full25"


class DsatG4ResetEffort2Policy(DsatGen31Policy):
    """Double weight on the air-side saturation term of the water reset: heat
    the water up when handlers are pinned, even at modest shortfalls."""

    RESET_EFFORT_WEIGHT = 2.0
    KEY = "g4_reset_effort2"


class DsatG4ResetTightPolicy(DsatGen31Policy):
    """Tighter reset follow: deadband 0.1 K instead of 0.25."""

    RESET_DEADBAND_K = 0.1
    KEY = "g4_reset_tight"


class DsatG4FPMidPolicy(DsatGen31Policy):
    """Forced-on AHUs escalate at mid-scale static pressure (-0.6 of range)
    instead of the full-scale -1.0: the 200 Pa minimum airflow cannot deliver
    reheat against stochastic convection; mid pressure is cheaper and rarely
    the binding constraint."""

    FORCED_PRESSURE = -0.6
    KEY = "g4_fp_06"


class DsatG4PrestartGatePolicy(DsatGen31Policy):
    """gen-3.1 chain + the occupant-EMA prestart gate: when the building has
    been empty for the last ~24 h (occupant mean < 0.5), skip the prestart
    burn entirely instead of heating an empty plant. Normal workdays are
    unchanged. gen-3.1 carries only the static hour gate."""

    OCC_24H_ALPHA = 1.0 - 1.0 / 288.0   # ~24-h EMA at 5-min steps
    KEY = "g4_prestart_gate"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._occ_ema = 0.0

    def _pi_hours(self, obs):
        if self._occ_idx is not None:
            occ = float(obs[self._occ_idx])
            self._occ_ema = (self.OCC_24H_ALPHA * self._occ_ema
                             + (1.0 - self.OCC_24H_ALPHA) * occ)
            if self._occ_ema < 0.5:
                hour, _ = self._get_hour_and_day(obs)
                if hour < self.morning_start_hour + 2:
                    return False
        return super()._pi_hours(obs)


class DsatG4Mild025Policy(DsatGen31Policy):
    """Milder supply for mixed-load handlers: MILD_SAT -0.4 -> -0.25. Deep
    supply into reheat-loaded air is the gas double-payment the mixedness
    gate exists to prevent; fairbanks pays it hardest."""

    MILD_SAT = -0.25
    KEY = "g4_mild025"


class DsatG4HeatGate08Policy(DsatGen31Policy):
    """Wider mild-SAT umbrella: heat share gate 0.05 -> 0.08, so slightly
    colder-climate handlers also classify as mixed and skip deep cooling."""

    HEAT_SHARE_GATE = 0.08
    KEY = "g4_heatgate08"


class DsatG4WPumpFracPolicy(DsatWaterPumpFracPolicy, DsatGen31Policy):
    """Pump head follows the water reset fraction instead of pinning full
    scale whenever any zone wants heat: fairbanks spends pump effort the
    zones mostly do not need."""

    KEY = "g4_wpump_frac"


class DsatG4WPumpIdleOffPolicy(DsatWaterPumpIdleOffPolicy, DsatGen31Policy):
    """Water pump (head + discrete) fully off when no zone wants heat: the
    PI skeleton pins the pump ON for all PI hours even at zero demand."""

    KEY = "g4_wpump_idleoff"


class DsatG4Prestart2hPolicy(DsatGen31Policy):
    """One extra prestart hour (PRESTART_H 1 -> 2): recover the building
    before wake at a gentler gas rate instead of a short hot sprint."""

    PRESTART_H = 2
    KEY = "g4_prestart2h"


class DsatG4Shift15Policy(DsatGen31Policy):
    """Deeper cheap-side zone target: TARGET_SHIFT 1.0 -> 1.5 K (down-only,
    cold outside). Zones ride closer to the low band edge while the PI
    holds, spending less water heat - the fairbanks gas axis."""

    TARGET_SHIFT_K = 1.5
    KEY = "g4_shift15"


class DsatG4Shift20Policy(DsatGen31Policy):
    """Aggressive twin of shift15: TARGET_SHIFT 2.0 K. The 0.4*(hi-lo) cap
    in _zone_target binds wherever the band is narrow."""

    TARGET_SHIFT_K = 2.0
    KEY = "g4_shift20"


class DsatG4Water08Policy(DsatGen31Policy):
    """Hot-water command scaled to 80% of range (WATER_SCALE 1.0 -> 0.8):
    cooler loop, cheaper gas, bet on reset demand covering the shortfall."""

    WATER_SCALE = 0.8
    KEY = "g4_water08"


class DsatG4NightDrop3Policy(DsatGen31Policy):
    """Deeper night heating setback: NIGHT_HEAT_DROP_K 2.0 -> 3.0. Fairbanks
    nights are the gas sink; rebet the morning catch-up absorbs the dip."""

    NIGHT_HEAT_DROP_K = 3.0
    KEY = "g4_nightdrop3"


class DsatG4Kp06Policy(DsatGen31Policy):
    """Gentler zone PI: KP 0.07 -> 0.06 (KI unchanged). Slower, smaller heat
    pulls; bet on fewer overshoot-reheat cycles paying the gas back."""

    KP = 0.06
    KEY = "g4_kp06"


class DsatG4Kp08Policy(DsatGen31Policy):
    """Faster zone PI: KP 0.07 -> 0.08. Bet on shorter catch-up sprints
    (smaller integrated shortfall) beating the extra pull depth."""

    KP = 0.08
    KEY = "g4_kp08"


class DsatG4HeatDamp085Policy(DsatGen31Policy):
    """Dampers follow heating demand at 85% gain instead of 1.0: less hot
    airflow per unit demand - the 'reheated air costs gas' lever."""

    HEAT_DAMPER_GAIN = 0.85
    KEY = "g4_hdamp085"


class DsatG4Water07Policy(DsatGen31Policy):
    """Escalation of the only clean round-3 winner: hot-water command scaled
    to 70% of range (water08 was -0.9%..-1.7% on all tested fairbanks)."""

    WATER_SCALE = 0.7
    KEY = "g4_w07"


class DsatG4M025W08Policy(DsatG4Mild025Policy):
    """Early combo of the two independent round-2/3 winners: MILD_SAT -0.25
    (hot-climate AHU SAT depth) stacked with WATER_SCALE 0.8 (loop head)."""

    WATER_SCALE = 0.8
    KEY = "g4_m25w08"


class DsatG4Kp05Policy(DsatGen31Policy):
    """KP gradient escalation: 0.05 (round-4: kp06 improved all five tested
    scenarios, kp08 worsened them)."""

    KP = 0.05
    KEY = "g4_kp05"


class DsatG4W06Policy(DsatGen31Policy):
    """WATER gradient escalation: 0.6 (round-4: w07 beat w08 on every
    tested scenario)."""

    WATER_SCALE = 0.6
    KEY = "g4_w06"


class DsatG4Kp06W07Policy(DsatGen31Policy):
    """Combo bracket: KP 0.06 + WATER_SCALE 0.7 (the two round-4 winners)."""

    KP = 0.06
    WATER_SCALE = 0.7
    KEY = "g4_kp06w07"


class DsatG4Kp06W08Policy(DsatGen31Policy):
    """Milder combo bracket: KP 0.06 + WATER_SCALE 0.8."""

    KP = 0.06
    WATER_SCALE = 0.8
    KEY = "g4_kp06w08"


class DsatG4M25W08Kp06Policy(DsatG4M025W08Policy):
    """Triple candidate: MILD_SAT -0.25 + WATER_SCALE 0.8 + KP 0.06 - the
    three independent winners stacked for the full-table gate check."""

    KP = 0.06
    KEY = "g4_m25w08kp06"


class DsatG4Kp04Policy(DsatGen31Policy):
    """KP gradient step 4: 0.04 (round-5: kp05 improved all five tested
    scenarios, regret included)."""

    KP = 0.04
    KEY = "g4_kp04"


class DsatG4Kp035Policy(DsatGen31Policy):
    """KP floor probe: 0.035 (round-6 still showed slope at 0.04 on fb1/fb8)."""

    KP = 0.035
    KEY = "g4_kp035"


class DsatG4Kp03Policy(DsatGen31Policy):
    """KP floor probe: 0.03. Comfort cliff expected somewhere below 0.04
    (morning catch-up is integration-only); fb2 is the sentinel."""

    KP = 0.03
    KEY = "g4_kp03"


class DsatG4M25W08Kp03Policy(DsatG4M025W08Policy):
    """Freeze candidate of the closed rule track: MILD_SAT -0.25 +
    WATER_SCALE 0.8 + KP 0.03. The exact 30-table was NOT run: the kp05
    variant measured -2.0% (gate is -5%), and the KP floor probes put the
    remaining slope at ~-0.3% more. Kept as the future teacher config for
    the parked NN/RL line - measure on the 5-grid (fb1,fb2,fb8,atl2,sg7)
    before any full-table dump."""

    KP = 0.03
    KEY = "g4_m25w08kp03"


class DsatG4Kp045Policy(DsatGen31Policy):
    """KP bracket-midpoint: 0.045, to map the curvature between 0.04 and 0.05."""

    KP = 0.045
    KEY = "g4_kp045"


class DsatG4Kp05W07Policy(DsatGen31Policy):
    """Double-deep combo: KP 0.05 + WATER_SCALE 0.7."""

    KP = 0.05
    WATER_SCALE = 0.7
    KEY = "g4_kp05w07"


class DsatG4M25W08Kp05Policy(DsatG4M025W08Policy):
    """Triple with the stronger PI: MILD_SAT -0.25 + WATER_SCALE 0.8 + KP 0.05."""

    KP = 0.05
    KEY = "g4_m25w08kp05"


class DsatG4Kp06Ki005Policy(DsatGen31Policy):
    """Classic low-P/high-I rebalance: KP 0.06 with KI 0.0035 -> 0.005 - as
    P shrinks the integrator carries the steady-state load overnight."""

    KP = 0.06
    KI = 0.005
    KEY = "g4_kp06ki005"


class DsatG4KiOaPolicy(DsatGen31Policy):
    """Outdoor-temperature KI schedule, round-6 data-driven split: a high
    integrator carried the deep-cold steady state (fb2 best), while warmth
    prefers the stock low KI (atlanta regressed +7% at KI 0.005). The split
    reads outdoor air temperature - a per-step observation, no identities."""

    KI_COLD = 0.005
    KI_WARM = 0.0035
    OAT_SPLIT_K = 273.15
    KEY = "g4_ki_oa"

    def _base_action(self, obs):
        if getattr(self, "_oat_any", None) is not None:
            try:
                oat = _outside_kelvin(self, float(obs[self._oat_any]))
                self.KI = self.KI_COLD if oat < self.OAT_SPLIT_K else self.KI_WARM
            except Exception:
                pass
        return super()._base_action(obs)
