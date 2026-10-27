# -*- coding: utf-8 -*-
"""THROWAWAY headroom probes (mode overrides). Delete after use."""

from smart_buildings.baselines.occ_gated4 import OccGated4Policy
from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class WeekendFloorProbe(OccGated4Policy):
    """occ_gated4, but cold weekends hold floor_heat instead of shutdown."""

    KEY = "probe_wfloor"

    def _decide(self, obs, occupied, mean_now):
        mode = super()._decide(obs, occupied, mean_now)
        if mode == "shutdown" and not occupied:
            hour, day = self._get_hour_and_day(obs)
            if day >= 5 and mean_now < 290.0:
                return "floor_heat"
        return mode


class EconProbe(OccGated4Policy):
    """occ_gated4 + economizer on forced-on AHUs (they skip the econ today).

    The base loop only calls _recirculation when the average-zone error is
    nonzero. Forced-on AHUs (per-zone need, average in band) keep the
    default recirc=1.0 even with free cooling outside. Recompute recirc
    from the FINAL (demand-driven) SAT for forced units.
    """

    KEY = "probe_econ"

    def _base_action(self, obs):
        action = super()._base_action(obs)
        cont = action["continuous_action"]
        hour, day = self._get_hour_and_day(obs)
        if day < 5 and self.morning_start_hour <= hour < self.evening_start_hour:
            low, high = self.comfort_temp_window
        else:
            low, high = self.eco_temp_window
        for i in getattr(self, "_forced", []):
            ctrl = self.ahu_controllers[i]
            if ctrl.get("recirc_idx") is None:
                continue
            u = float(cont[ctrl["temp_idx"]])
            cont[ctrl["recirc_idx"]] = self._recirculation(
                obs, ctrl, u, low, high)
        return action


class MarginProbe(OccGated4Policy):
    """occ_gated4 with prestart/floor margins from $M_CAP/$M_SAFE/$M_MARGIN."""

    KEY = "probe_margin"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self.FLOOR_CAPABILITY = float(_os.environ.get("M_CAP", "0.7"))
        self.PRESTART_SAFETY = float(_os.environ.get("M_SAFE", "1.3"))
        self.PRESTART_MARGIN_STEPS = int(_os.environ.get("M_MARGIN", "6"))


class SATProbe(OccGated4Policy):
    """occ_gated4 with AHU supply-temp biased by $SAT_BIAS (normalized)."""

    KEY = "probe_sat"

    def get_action(self, obs):
        import os as _os
        import numpy as _np
        action = super().get_action(obs)
        b = float(_os.environ.get("SAT_BIAS", "0.0"))
        if b != 0.0:
            cont = action["continuous_action"]
            for ctrl in self.ahu_controllers:
                i = ctrl["temp_idx"]
                cont[i] = _np.float32(_np.clip(float(cont[i]) + b, -1.0, 1.0))
        return action


class TodShiftProbe(OccGated4Policy):
    """occ_gated4 with time-of-day cooling offsets (load shifting).

    Cool target -1K at 00-10h (pre-cool on cheap/clean power),
    +1K at 12-19h (coast through the price/carbon peak), else +0.
    """

    KEY = "probe_tod"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._base_lo, self._base_hi = self.comfort_temp_window

    def get_action(self, obs):
        hour, _ = self._get_hour_and_day(obs)
        if 0 <= hour < 10:
            dc = -1.0
        elif 12 <= hour < 19:
            dc = 1.0
        else:
            dc = 0.0
        self.comfort_temp_window = (self._base_lo, self._base_hi + dc)
        return super().get_action(obs)


class GuardProbe(OccGated4Policy):
    """occ_gated4 + per-zone comfort guard (transfer insurance).

    Mean-based shutdown/floor decisions can strand a single cold zone on
    unseen buildings, and the reward's quadratic unbounded comfort term
    lets one stranded zone dominate an episode. While occupied, any zone
    more than GUARD_K outside the comfort band forces normal conditioning
    instead of shutdown/floor modes. Hysteresis on release.
    """

    KEY = "probe_guard"
    GUARD_K = 3.0
    GUARD_RELEASE_K = 2.0

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._guard_latched = False

    def _max_zone_deviation(self, obs) -> float:
        lo, hi = self.comfort_temp_window
        worst = 0.0
        for j in self._all_zone_obs:
            t = self._denormalize_zone_temp(obs[j])
            if t < lo:
                worst = max(worst, lo - t)
            elif t > hi:
                worst = max(worst, t - hi)
        return float(worst)

    def get_action(self, obs):
        occupied = self._occupied(obs)
        dev = self._max_zone_deviation(obs)
        if occupied and dev > self.GUARD_K:
            self._guard_latched = True
        elif dev < self.GUARD_RELEASE_K:
            self._guard_latched = False
        if self._guard_latched and occupied:
            mean_now = self._mean_zone_temp(obs)
            self._per_step(obs, mean_now)
            self._prestart_active = False
            self._floor_mode = None
            self._shutdown = False
            return self._chain_action(obs)
        return super().get_action(obs)


class PressureTrimRespondProbe(OccGated4Policy):
    """occ_gated4 + trim-and-respond static pressure (ASHRAE G36 style).

    pressmin pins every running AHU to -1.0. On big-zone buildings the
    starved airflow cannot carry enough VAV-reheat heat (discharge capped
    at 308.15 K), so zones freeze with dampers wide open (xfer_3 January:
    comfort -0.06). Respond: any zone out of band with a saturated damper
    steps its AHU pressure up fast; trim: all dampers closed steps it down
    slowly toward pressmin. Only acts when occupied or prestarting, so the
    overnight floor keeps minimum pressure.
    """

    KEY = "probe_ptr"

    def __init__(self, building_meta_data, action_spec):
        import os as _os
        super().__init__(building_meta_data, action_spec)
        self._ptr_up = float(_os.environ.get("PTR_UP", "0.25"))
        self._ptr_down = float(_os.environ.get("PTR_DOWN", "0.03"))
        self._ptr_max = float(_os.environ.get("PTR_MAX", "0.5"))
        self._ptr_dsat = float(_os.environ.get("PTR_DSAT", "0.85"))
        self._ptr_db = float(_os.environ.get("PTR_DB", "0.5"))
        self._ptr_db_slope = float(_os.environ.get("PTR_DB_SLOPE", "0.0"))
        self._ptr_db_ref = float(_os.environ.get("PTR_DB_REF", "10.0"))
        self._ptr_persist = int(_os.environ.get("PTR_PERSIST", "1"))
        self._ptr_occ_only = int(_os.environ.get("PTR_OCC_ONLY", "0"))
        self._ptr_p = [-1.0] * len(self.ahu_controllers)
        self._ptr_nstarve = [0] * len(self.ahu_controllers)

    def _base_action(self, obs):
        import numpy as _np
        action = super()._base_action(obs)
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        _occ = self._occupied(obs)
        active = _occ if self._ptr_occ_only else (_occ or getattr(self, "_prestart_active", False))
        lo, hi = self.comfort_temp_window
        # Air is cheap near the band, ruinous at large |OAT-band| (ventilation
        # load): scale the excursion deadband with the outside delta so deep
        # cold/heat only answers deep bleeds.
        db_eff = self._ptr_db
        if self._ptr_db_slope > 0 and getattr(self, "_oat_any", None) is not None:
            try:
                from smart_buildings.baselines.zone_pi import _outside_kelvin as _ok
                oat = _ok(self, float(obs[self._oat_any]))
                dt = abs(oat - 0.5 * (lo + hi))
                db_eff = self._ptr_db + self._ptr_db_slope * max(0.0, dt - self._ptr_db_ref)
            except (TypeError, ValueError):
                pass
        for i, ctrl in enumerate(self.ahu_controllers):
            pidx = ctrl["pressure_idx"]
            if pidx is None:
                continue
            if not disc[ctrl["run_discrete_idx"]]:
                self._ptr_p[i] = -1.0
                self._ptr_nstarve[i] = 0
                continue
            if not active:
                continue
            starved = False
            all_closed = True
            for vav_id, obs_idx, _ in ctrl["zones"]:
                aidx = self.zone_action_indices.get(vav_id, {}).get(
                    "supply_air_damper_percentage_command")
                if aidx is None:
                    continue
                d = (float(cont[aidx]) + 1.0) * 0.5
                if d >= 0.5:
                    all_closed = False
                t = self._denormalize_zone_temp(obs[obs_idx])
                if d > self._ptr_dsat and (t < lo - db_eff or t > hi + db_eff):
                    starved = True
            if starved:
                self._ptr_nstarve[i] += 1
            else:
                self._ptr_nstarve[i] = 0
            if self._ptr_nstarve[i] >= self._ptr_persist:
                self._ptr_p[i] = min(self._ptr_max, self._ptr_p[i] + self._ptr_up)
            elif all_closed:
                self._ptr_p[i] = max(-1.0, self._ptr_p[i] - self._ptr_down)
            cont[pidx] = _np.float32(self._ptr_p[i])
        return action


class SetpointProbe(OccGated4Policy):
    """occ_gated4 with comfort-window shifted by $SP_DH/$SP_DC kelvin."""

    KEY = "probe_sp"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        dh = float(_os.environ.get("SP_DH", "0.0"))
        dc = float(_os.environ.get("SP_DC", "0.0"))
        lo, hi = self.comfort_temp_window
        self.comfort_temp_window = (lo + dh, hi + dc)


class PtrFastTrimProbe(OccGated7Policy):
    """occ7 + fast trim-down when every zone is satisfied (P-a).

    Oracle (CEM const-offset, xfer_3 January Tue/Wed) cuts water/air with
    zero comfort at stake: trim-respond holds pressure up after the
    excursion is gone. This probe trims at $PA_DOWN (default 0.10) whenever
    no zone is outside band +/- $PA_SAT_DB (default 0.25K), even with
    dampers mid-open. $PA_DOWN=0 reproduces occ7 exactly.
    """

    KEY = "probe_ptr_pa"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._pa_sat_db = float(_os.environ.get("PA_SAT_DB", "0.25"))
        self._pa_down = float(_os.environ.get("PA_DOWN", "0.10"))

    def _base_action(self, obs):
        import numpy as _np
        action = super(OccGated7Policy, self)._base_action(obs)
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        active = self._occupied(obs)
        lo, hi = self.comfort_temp_window
        db_eff = self.PTR_DB
        if self.PTR_DB_SLOPE > 0 and getattr(self, "_oat_any", None) is not None:
            try:
                oat = _outside_kelvin(self, float(obs[self._oat_any]))
                dt = abs(oat - 0.5 * (lo + hi))
                db_eff = self.PTR_DB + self.PTR_DB_SLOPE * max(0.0, dt - self.PTR_DB_REF)
            except (TypeError, ValueError):
                pass
        for i, ctrl in enumerate(self.ahu_controllers):
            pidx = ctrl["pressure_idx"]
            if pidx is None:
                continue
            if not disc[ctrl["run_discrete_idx"]]:
                self._ptr_p[i] = -1.0
                self._ptr_nstarve[i] = 0
                continue
            if not active:
                continue
            starved = False
            all_closed = True
            any_out = False
            for vav_id, obs_idx, _ in ctrl["zones"]:
                aidx = self.zone_action_indices.get(vav_id, {}).get(
                    "supply_air_damper_percentage_command")
                if aidx is None:
                    continue
                d = (float(cont[aidx]) + 1.0) * 0.5
                if d >= 0.5:
                    all_closed = False
                t = self._denormalize_zone_temp(obs[obs_idx])
                if d > self.PTR_DSAT and (t < lo - db_eff or t > hi + db_eff):
                    starved = True
                if t < lo - self._pa_sat_db or t > hi + self._pa_sat_db:
                    any_out = True
            if starved:
                self._ptr_nstarve[i] += 1
            else:
                self._ptr_nstarve[i] = 0
            if self._ptr_nstarve[i] >= self.PTR_PERSIST:
                self._ptr_p[i] = min(self.PTR_MAX, self._ptr_p[i] + self.PTR_UP)
            elif all_closed:
                self._ptr_p[i] = max(-1.0, self._ptr_p[i] - self.PTR_DOWN)
            elif not any_out:
                self._ptr_p[i] = max(-1.0, self._ptr_p[i] - self._pa_down)
            cont[pidx] = _np.float32(self._ptr_p[i])
        return action


class PtrPropKickProbe(OccGated7Policy):
    """occ7 + respond kick proportional to excursion depth (P-b).

    Oracle (xfer_3 January Thu, real bleed) pushes harder than +0.25/step
    where it matters: kick = UP * (1 + $PB_PROP * max excursion beyond
    db among open-damper zones). $PB_PROP=0 reproduces occ7 exactly.
    """

    KEY = "probe_ptr_pb"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._pb_prop = float(_os.environ.get("PB_PROP", "0.5"))

    def _base_action(self, obs):
        import numpy as _np
        action = super(OccGated7Policy, self)._base_action(obs)
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        active = self._occupied(obs)
        lo, hi = self.comfort_temp_window
        db_eff = self.PTR_DB
        if self.PTR_DB_SLOPE > 0 and getattr(self, "_oat_any", None) is not None:
            try:
                oat = _outside_kelvin(self, float(obs[self._oat_any]))
                dt = abs(oat - 0.5 * (lo + hi))
                db_eff = self.PTR_DB + self.PTR_DB_SLOPE * max(0.0, dt - self.PTR_DB_REF)
            except (TypeError, ValueError):
                pass
        for i, ctrl in enumerate(self.ahu_controllers):
            pidx = ctrl["pressure_idx"]
            if pidx is None:
                continue
            if not disc[ctrl["run_discrete_idx"]]:
                self._ptr_p[i] = -1.0
                self._ptr_nstarve[i] = 0
                continue
            if not active:
                continue
            starved = False
            all_closed = True
            exc = 0.0
            for vav_id, obs_idx, _ in ctrl["zones"]:
                aidx = self.zone_action_indices.get(vav_id, {}).get(
                    "supply_air_damper_percentage_command")
                if aidx is None:
                    continue
                d = (float(cont[aidx]) + 1.0) * 0.5
                if d >= 0.5:
                    all_closed = False
                t = self._denormalize_zone_temp(obs[obs_idx])
                if d > self.PTR_DSAT and (t < lo - db_eff or t > hi + db_eff):
                    starved = True
                    exc = max(exc, lo - db_eff - t, t - hi - db_eff)
            if starved:
                self._ptr_nstarve[i] += 1
            else:
                self._ptr_nstarve[i] = 0
            if self._ptr_nstarve[i] >= self.PTR_PERSIST:
                # float() firewall: exc can be np.float32 (zone temps), and
                # NEP-50 weak promotion would round _ptr_p to float32.
                kick = float(self.PTR_UP * (1.0 + self._pb_prop * float(exc)))
                self._ptr_p[i] = min(self.PTR_MAX, self._ptr_p[i] + kick)
            elif all_closed:
                self._ptr_p[i] = max(-1.0, self._ptr_p[i] - self.PTR_DOWN)
            cont[pidx] = _np.float32(self._ptr_p[i])
        return action


class PtrHighCeilProbe(OccGated7Policy):
    """occ7 + higher pressure ceiling (P-c).

    Oracle pushes AHU_P4 +0.34 past occ7's ceiling (PTR_MAX=0.0) in the
    extreme state: ceiling, not trigger, binds there. $PC_MAX (default
    0.5); $PC_MAX=0.0 reproduces occ7 exactly.
    """

    KEY = "probe_ptr_pc"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._pc_max = float(_os.environ.get("PC_MAX", "0.5"))

    def _base_action(self, obs):
        import numpy as _np
        action = super(OccGated7Policy, self)._base_action(obs)
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        active = self._occupied(obs)
        lo, hi = self.comfort_temp_window
        db_eff = self.PTR_DB
        if self.PTR_DB_SLOPE > 0 and getattr(self, "_oat_any", None) is not None:
            try:
                oat = _outside_kelvin(self, float(obs[self._oat_any]))
                dt = abs(oat - 0.5 * (lo + hi))
                db_eff = self.PTR_DB + self.PTR_DB_SLOPE * max(0.0, dt - self.PTR_DB_REF)
            except (TypeError, ValueError):
                pass
        for i, ctrl in enumerate(self.ahu_controllers):
            pidx = ctrl["pressure_idx"]
            if pidx is None:
                continue
            if not disc[ctrl["run_discrete_idx"]]:
                self._ptr_p[i] = -1.0
                self._ptr_nstarve[i] = 0
                continue
            if not active:
                continue
            starved = False
            all_closed = True
            for vav_id, obs_idx, _ in ctrl["zones"]:
                aidx = self.zone_action_indices.get(vav_id, {}).get(
                    "supply_air_damper_percentage_command")
                if aidx is None:
                    continue
                d = (float(cont[aidx]) + 1.0) * 0.5
                if d >= 0.5:
                    all_closed = False
                t = self._denormalize_zone_temp(obs[obs_idx])
                if d > self.PTR_DSAT and (t < lo - db_eff or t > hi + db_eff):
                    starved = True
            if starved:
                self._ptr_nstarve[i] += 1
            else:
                self._ptr_nstarve[i] = 0
            if self._ptr_nstarve[i] >= self.PTR_PERSIST:
                self._ptr_p[i] = min(self._pc_max, self._ptr_p[i] + self.PTR_UP)
            elif all_closed:
                self._ptr_p[i] = max(-1.0, self._ptr_p[i] - self.PTR_DOWN)
            cont[pidx] = _np.float32(self._ptr_p[i])
        return action
