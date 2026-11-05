# -*- coding: utf-8 -*-
"""THROWAWAY headroom probes (mode overrides). Delete after use."""

import numpy as np

from smart_buildings.baselines.occ_gated4 import OccGated4Policy
from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.slot_hybrid import SlotHybridPolicy
from smart_buildings.baselines.occ_coolshift7 import CoolShiftMixin
from smart_buildings.baselines.occ_econdbhhot9 import OccEconDbhHot9Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class SkynetResidualProbe:
    """Local-only measurement probe: shipped residual weights on the zt_off base.

    Exact reproduction of the dataset policy (skynet.pt@5001216 over ZTPolicy
    via make_floor controller zt_off). Local measurement only, not for
    submission: weights live outside the repo. Throwaway per conventions.
    """

    KEY = "probe_skynet"

    WEIGHTS = "/tmp/opencode/sb-rl/skynet.pt"
    PKGDIR = "/tmp/pkg"

    def __init__(self, building_meta_data, action_spec):
        import sys as _sys
        if self.PKGDIR not in _sys.path:
            _sys.path.insert(0, self.PKGDIR)
        from skynet.policy import load as _load
        from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
        from smart_buildings.evaluate import validate_action as _validate
        net, params, scale, latch = _load(self.WEIGHTS)
        from skynet.policy import SkynetPolicy as _Skynet
        self._pol = _Skynet(
            building_meta_data, action_spec, net, params, scale,
            PIDWaterResetPolicy, _validate, latch)

    @property
    def name(self):
        return self.KEY

    def get_action(self, obs):
        return self._pol.get_action(obs)


def any_cold(floor, ix, obs, k):
    """True if any zone sits more than k below its band low edge.

    Read-only with respect to the floor policy (band accessors are pure
    reads). Shared by the probe and the rollout driver so training sees
    exactly the gating deployment applies.
    """
    import numpy as _np
    zr = _np.asarray(obs, dtype=_np.float64)[ix.zone_obs]
    temp_k = zr[:, 0] * ix.t_std + ix.t_mean
    lo_c, hi_c = (float(x) for x in floor.comfort_temp_window)
    for i, vav in enumerate(ix.zone_vav):
        lo, hi = floor._zone_band(obs, vav, lo_c, hi_c)
        if hi - lo > (hi_c - lo_c) + 1e-6:
            lo, hi = lo_c, hi_c
        if float(temp_k[i]) < lo - k:
            return True
    return False


def residual_free(policy, ix, obs, cold_k):
    """True iff the residual acts this step (our `free` flag, plan F/D3).

    Mirrors Skynet's `floor.free` (PI-hours gating: off-hours => rule
    alone) plus our two extra gates: occ shutdown/floor modes and the
    cold gate. `cold_k=None` (or < 0) disables the cold term, for
    scale-0 identity runs where the gate would be a no-op anyway.
    Single source of truth shared by the probe and the rollout, so
    training sees exactly the gating deployment applies.
    """
    if getattr(policy, "_shutdown", False):
        return False
    if getattr(policy, "_floor_mode", None) is not None:
        return False
    if (cold_k is not None and cold_k >= 0.0
            and any_cold(policy, ix, obs, cold_k)):
        return False
    return True


class ExpandedResidualProbe(OccGated7Policy):
    """occ_gated7 + expanded residual harness (stateful inputs + target head).

    Zero-residual check probe for the Online line: the surgery-expanded
    FlatNets (tools/skynet_surgery.py::expand_net, loaded from skynet.pt)
    reads 23-wide zone features (21 base + true loop err + integral
    saturation) and its corrections are composed with RESIDUAL_SCALE.
    Switches stay frozen on the base (Bernoulli heads are a later step);
    the target head is dead by construction (the occ chain has no
    target_offset effector yet). With RESIDUAL_SCALE = 0.0 the action is
    bit-identical to occ_gated7 while exercising the full harness.
    Live corrections use scale 1.0. Local measurement only.
    """

    KEY = "probe_xres"
    WEIGHTS = "/tmp/opencode/sb-rl/skynet.pt"
    RESIDUAL_SCALE = 0.0
    # Cold gate (K below the band low edge): while any zone is this cold,
    # return the base action untouched. Measured failure it fixes: the
    # residual under-heated cold buildings to save gas (xfer_1/minneapolis_1
    # comfort regret 5.2x for -8% gas). One-sided by construction: mild
    # regimes keep the learned corrections, cold keeps the base.
    COLD_GATE_K = 1.0

    def _load_expanded(self):
        import os as _os
        import sys as _sys
        _surg = (_sys.modules.get("skynet_surgery")
                 or _sys.modules.get("tools.skynet_surgery"))
        from skynet.policy import load as _load
        path = _os.environ.get("SKYNET_WEIGHTS", self.WEIGHTS)
        ck = __import__("torch").load(path, map_location="cpu",
                                      weights_only=False)
        if "zone.0.adapter.weight" in ck.get("model", {}):
            net, _rep = _surg.expand_net_from_ckpt(ck)
            net.eval()
            return net
        net, _params, _scale, _latch = _load(path)
        net, _rep = _surg.expand_net(net)
        net.eval()
        return net

    def __init__(self, building_meta_data, action_spec):
        import os as _os
        self.RESIDUAL_SCALE = float(
            _os.environ.get("SKYNET_SCALE", self.RESIDUAL_SCALE))
        from pathlib import Path as _Path
        super().__init__(building_meta_data, action_spec)
        import sys as _sys
        if "/tmp/pkg" not in _sys.path:
            _sys.path.insert(0, "/tmp/pkg")
        import importlib.util as _ilu
        _surg = (_Path(__file__).resolve().parents[2]
                 / "tools" / "skynet_surgery.py")
        _spec = _ilu.spec_from_file_location("skynet_surgery", _surg)
        _mod = _ilu.module_from_spec(_spec)
        _sys.modules["skynet_surgery"] = _mod
        _spec.loader.exec_module(_mod)
        import torch as _torch
        try:
            _torch.set_num_threads(1)
            _torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # ThreadPool workers start parallel work before policies
        self._torch = _torch
        self._xnet = self._load_expanded()
        from skynet.features import BuildingIndex as _BI
        self._ix = _BI(building_meta_data)

    def _expanded_zone_cols(self, obs):
        import sys as _sys
        _mod = _sys.modules.get("skynet_surgery")
        return _mod.expanded_zone_cols(self, self._ix, obs)

    def _any_cold(self, obs, k):
        return any_cold(self, self._ix, obs, k)

    def get_action(self, obs):
        import numpy as _np
        action = super().get_action(obs)
        cold = (self.COLD_GATE_K
                if self.RESIDUAL_SCALE != 0.0 else None)
        if not residual_free(self, self._ix, obs, cold):
            return action
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        zf, af, pf = self._ix.features(obs, cont, disc)
        extra = self._expanded_zone_cols(obs)
        z23 = _np.concatenate([zf, extra], axis=1).astype(_np.float32)
        t = self._torch.from_numpy
        with self._torch.no_grad():
            (mz, ma, mp), _v = self._xnet(
                t(z23)[None], self._torch.ones(1, len(z23)),
                t(af)[None], self._torch.ones(1, len(af)),
                t(pf)[None])
            mz, ma, mp = mz[0].numpy(), ma[0].numpy(), mp[0].numpy()
        cont2, _ = self._ix.compose(
            cont, disc, mz, ma, mp, self.RESIDUAL_SCALE)
        return {"continuous_action": cont2,
                "discrete_action": disc}


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


class Ptr9Up15Probe(OccEconDbhHot9Policy):
    """J2: occ9 with gentler PTR respond (UP 0.25 -> 0.15)."""

    KEY = "probe_ptr9_up15"
    PTR_UP = 0.15


class Ptr9Db20Probe(OccEconDbhHot9Policy):
    """J2: occ9 with wider PTR deadband (DB 1.0 -> 2.0)."""

    KEY = "probe_ptr9_db20"
    PTR_DB = 2.0


class Ptr9Persist8Probe(OccEconDbhHot9Policy):
    """J2: occ9 with slower PTR trigger (PERSIST 4 -> 8)."""

    KEY = "probe_ptr9_persist8"
    PTR_PERSIST = 8


class Ptr9Up35Probe(OccEconDbhHot9Policy):
    """J2: occ9 with stronger PTR respond (UP 0.25 -> 0.35, sign check)."""

    KEY = "probe_ptr9_up35"
    PTR_UP = 0.35


class Hopt9SatU07Probe(OccEconDbhHot9Policy):
    """J2: occ9 with fewer heatopt-saturated zones (SAT_U -0.9 -> -0.7)."""

    KEY = "probe_hopt9_satu07"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._hopt_sat_u = -0.7


class Dbh9Db10Probe(OccEconDbhHot9Policy):
    """J3: occ9 with narrower heating deadband (DB 0.15 -> 0.10 K)."""

    KEY = "probe_dbh9_db10"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._dbh_db = 0.10


class Dbh9Leak100Probe(OccEconDbhHot9Policy):
    """J3: occ9 with no integral leak (LEAK 0.97 -> 1.00)."""

    KEY = "probe_dbh9_leak100"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._dbh_leak = 1.00


class Dbh9LeakNightProbe(OccEconDbhHot9Policy):
    """J3: occ9 with the 0.97 leak only at 00-06, persist otherwise."""

    KEY = "probe_dbh9_leaknight"

    def _zone_err(self, obs, temp, lo, hi):
        self._leak_hour = self._get_hour_and_day(obs)[0]
        return super()._zone_err(obs, temp, lo, hi)

    def _zone_integral_update(self, err, integral):
        if self._dbh_on and err == 0.0:
            h = getattr(self, "_leak_hour", 12)
            leak = self._dbh_leak if 0 <= h < 6 else 1.0
            return integral * leak
        return super()._zone_integral_update(err, integral)


class Csh9Probe(CoolShiftMixin, OccEconDbhHot9Policy):
    """J4: occ9 + cooling-side warm target shift (0.3K)."""

    KEY = "probe_csh9"


class Kp069Probe(OccEconDbhHot9Policy):
    """J4: occ9 with gentler zone PI (KP 0.07 -> 0.06)."""

    KEY = "probe_kp069"
    KP = 0.06


class Csh9Kp06Probe(CoolShiftMixin, OccEconDbhHot9Policy):
    """J4: occ9 + coolshift + gentler PI (full stack)."""

    KEY = "probe_csh9_kp06"
    KP = 0.06


# J5 menu: fixed PRESTART_H is dead on the occ path (_decide latches prestart
# adaptively and _prestart_active forces the PI window open), so the menu
# varies the live knob PRESTART_MARGIN_STEPS (shipped 6 = +30 min).


class Ps9H0Probe(OccEconDbhHot9Policy):
    """J5: occ9 with zero prestart margin (MARGIN 6 -> 0)."""

    KEY = "probe_ps9_h0"
    PRESTART_MARGIN_STEPS = 0


class Ps9H1Probe(OccEconDbhHot9Policy):
    """J5: occ9 with shipped margin (MARGIN 6, control)."""

    KEY = "probe_ps9_h1"
    PRESTART_MARGIN_STEPS = 6


class Ps9H2Probe(OccEconDbhHot9Policy):
    """J5: occ9 with doubled margin (MARGIN 6 -> 12, +1h)."""

    KEY = "probe_ps9_h2"
    PRESTART_MARGIN_STEPS = 12


class Ps9H3Probe(OccEconDbhHot9Policy):
    """J5: occ9 with 4x margin (MARGIN 6 -> 24, +2h)."""

    KEY = "probe_ps9_h3"
    PRESTART_MARGIN_STEPS = 24


# P20-8 R3a oracle menu: one live branch per arm (J5 fixed-const pattern).
# Menu v1 (COLD_SAT / HOPT_P_MAX / ECON_FOUT_BIAS) died in liveness:
# demand-SAT is overwritten by MixedSat, heatopt never raises pressure
# on b8, econ-gated branches never fire. Menu v2 (FORCED_SAT /
# FORCED_PRESSURE / raw recirc bias) died too: forced branches are also
# overwritten by MixedSat, and raw recirc +/-0.15 is metric-dead (sim
# clamps recirc at 0.70; cont differs but reward is bit-identical). Menu
# v3 roots each dimension onto a branch with fire-count proof on b8x5.
# Neutral == occ9 bit-for-bit (sim parity, not an arm run): the occ9
# control leg comes from the D4 train-30 CSVs (shipped bill, seed 17321,
# deterministic replay verified by cmp before the suite).


class R3aSatProbe(OccEconDbhHot9Policy):
    """R3a: occ9 with deepened MixedSat mild branch (MILD_SAT -0.4 -> -1.0).

    DEEP_SAT writes land only on stopped AHUs (run1_frac 0.00 -- dead
    channel); MILD_SAT writes land on running AHUs (run1_frac 1.00).
    """

    KEY = "probe_r3a_sat"
    MILD_SAT = -1.0


class _PressureFloorMixin:
    """Static-pressure floor applied above the occ9 chain."""

    PRESSURE_FLOOR = -1.0

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.PRESSURE_FLOOR > -1.0:
            cont = action["continuous_action"]
            for ctrl in self.ahu_controllers:
                p_idx = ctrl.get("pressure_idx")
                if p_idx is not None:
                    cont[p_idx] = np.float32(max(
                        float(cont[p_idx]), self.PRESSURE_FLOOR))
        return action


class R3aPcapProbe(_PressureFloorMixin, OccEconDbhHot9Policy):
    """R3a: occ9 with raised pressure floor (-1.0 -> -0.6).

    A cap is unrepresentable on b8x5 (pressure pinned at minimum all
    episode: pressmin + dead T&R), so the pressure dimension enters as
    a floor lift (J6-mixin pattern above the chain).
    """

    KEY = "probe_r3a_pcap"
    PRESSURE_FLOOR = -0.6


class R3aRecProbe(OccEconDbhHot9Policy):
    """R3a: occ9 with capped econ outside-air fraction (clip-hi 1.0 -> 0.7)."""

    KEY = "probe_r3a_rec"
    ECON_FOUT_HI = 0.7


class R3aHwsProbe(OccEconDbhHot9Policy):
    """R3a: occ9 with cooler HWS (WATER_SCALE 1.0 -> 0.7)."""

    KEY = "probe_r3a_hws"
    WATER_SCALE = 0.7


# P20-7 N1.1: danielwuu (0.01061 honest) pieces on occ9. Ported from
# /tmp/kaggle_nb/danielwuu_cpu-b-*/policy.py (frozen archive, sha ok):
# (a) OFF-candidate (_choose_ahu: run AHU only if free-drift prediction
# leaves the band), (b) pressure = 2000 x fraction^2 reset, (c) water
# 345K/310K heat-recency hysteresis (24 steps). Each piece is one mixin
# (neutral via _ON flag) + single probe + the N1StackProbe combo.


class N1OffMixin:
    """danielwuu (a): per-AHU OFF when all group zones sit deep in band."""

    N1_OFF_ON = 1
    N1_OFF_MARGIN = 0.5

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self.N1_OFF_ON:
            return action
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        lo, hi = self.comfort_temp_window
        for ctrl in self.ahu_controllers:
            if not disc[ctrl["run_discrete_idx"]]:
                continue
            ok = True
            for _, obs_idx, _ in ctrl["zones"]:
                try:
                    t = self._denormalize_zone_temp(obs[obs_idx])
                except (TypeError, ValueError):
                    ok = False
                    break
                if not (lo + self.N1_OFF_MARGIN < t < hi - self.N1_OFF_MARGIN):
                    ok = False
                    break
            if ok:
                disc[ctrl["run_discrete_idx"]] = 0
                for vav_id, _, _ in ctrl["zones"]:
                    aidx = self.zone_action_indices.get(vav_id, {}).get(
                        "supply_air_damper_percentage_command")
                    if aidx is not None:
                        cont[aidx] = np.float32(2.0 * self.MIN_DAMPER - 1.0)
        return action


class N1OffProbe(N1OffMixin, OccEconDbhHot9Policy):
    """N1.1: occ9 + danielwuu OFF-candidate."""

    KEY = "probe_n1_off"


class N1P2Mixin:
    """danielwuu (b): per-AHU pressure = 2000 x max-damper-fraction^exp."""

    N1_P2_ON = 1
    N1_P2_EXP = 2.0
    N1_P2_MIN_PA = 200.0

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self.N1_P2_ON:
            return action
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        for ctrl in self.ahu_controllers:
            pidx = ctrl.get("pressure_idx")
            if pidx is None or not disc[ctrl["run_discrete_idx"]]:
                continue
            fmax = 0.0
            for vav_id, _, _ in ctrl["zones"]:
                aidx = self.zone_action_indices.get(vav_id, {}).get(
                    "supply_air_damper_percentage_command")
                if aidx is not None:
                    fmax = max(fmax, (float(cont[aidx]) + 1.0) * 0.5)
            p = min(2000.0, max(self.N1_P2_MIN_PA,
                                2000.0 * fmax ** self.N1_P2_EXP))
            cont[pidx] = np.float32(p / 1000.0 - 1.0)
        return action


class N1P2Probe(N1P2Mixin, OccEconDbhHot9Policy):
    """N1.1: occ9 + danielwuu pressure fraction^2 reset."""

    KEY = "probe_n1_p2"


class N1HystMixin:
    """danielwuu (c): HWS 345K/310K heat-recency hysteresis."""

    N1_HYST_ON = 1
    N1_HYST_WINDOW = 24
    N1_HYST_MARGIN = 0.0

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._n1_last_heat = -10 ** 9
        self._n1_step = 0

    def _base_action(self, obs):
        action = super()._base_action(obs)
        self._n1_step += 1
        if not self.N1_HYST_ON or self.water_temp_idx is None:
            return action
        lo, _ = self.comfort_temp_window
        for ctrl in self.ahu_controllers:
            for _, obs_idx, _ in ctrl["zones"]:
                try:
                    t = self._denormalize_zone_temp(obs[obs_idx])
                except (TypeError, ValueError):
                    continue
                if t < lo + self.N1_HYST_MARGIN:
                    self._n1_last_heat = self._n1_step
                    break
            else:
                continue
            break
        recent = (self._n1_step - self._n1_last_heat) < self.N1_HYST_WINDOW
        cont = action["continuous_action"]
        cont[self.water_temp_idx] = np.float32(0.75 if recent else -1.0)
        return action


class N1HystProbe(N1HystMixin, OccEconDbhHot9Policy):
    """N1.1: occ9 + danielwuu water hysteresis."""

    KEY = "probe_n1_hyst"


class N1StackProbe(N1OffMixin, N1P2Mixin, N1HystMixin, OccEconDbhHot9Policy):
    """N1.1: occ9 + all three danielwuu pieces (BO-tuned stack)."""

    KEY = "probe_n1_stack"


class N1StackTunedProbe(N1StackProbe):
    """N1.1: stack at BO-best (it04: margin 0.5408, exp 2.6692, win 71).

    Fixed consts from the tools/n1bo.py run (20 iters, best_sum
    0.116293 over the g4bo 5-scenario honest screen). P20-9 finals arm.
    """

    KEY = "probe_n1_tuned"
    N1_OFF_MARGIN = 0.5408
    N1_P2_EXP = 2.6692
    N1_HYST_WINDOW = 71


class _ZoneSideShiftMixin:
    """J6: rule-side in-band zone-target shift (their head analogue).

    Moves the effective PI target toward the zone temp by up to
    ZSS_SHIFT (in-band temps only): heat effort falls (target down),
    cooling effort falls (target up). Equivalent to a symmetric
    deadband of width ZSS_SHIFT on the inherited err; the implied
    target stays between temp and the old target, hence in-band.
    Never flips the err sign. ZSS_COOL_ONLY isolates the cooling
    half (the heating half duplicates dbh).
    """

    ZSS_SHIFT = 0.5
    ZSS_COOL_ONLY = False

    def _zone_err(self, obs, temp, lo, hi):
        err = super()._zone_err(obs, temp, lo, hi)
        temp = float(temp)
        if not (lo <= temp <= hi) or err == 0.0:
            return err
        if self.ZSS_COOL_ONLY and err < 0.0:
            return err
        s = min(self.ZSS_SHIFT, abs(err))
        return err - s if err > 0.0 else err + s


class Zss05Probe(_ZoneSideShiftMixin, OccEconDbhHot9Policy):
    """J6: occ9 + symmetric in-band shift 0.5K."""

    KEY = "probe_zss05"
    ZSS_SHIFT = 0.5


class Zss10Probe(_ZoneSideShiftMixin, OccEconDbhHot9Policy):
    """J6: occ9 + symmetric in-band shift 1.0K (their magnitude)."""

    KEY = "probe_zss10"
    ZSS_SHIFT = 1.0


class Zssc05Probe(_ZoneSideShiftMixin, OccEconDbhHot9Policy):
    """J6: occ9 + cooling-side-only shift 0.5K."""

    KEY = "probe_zssc05"
    ZSS_SHIFT = 0.5
    ZSS_COOL_ONLY = True


class _SlotHybrid7Mixin:
    """MoE T3: SlotHybrid heads under OccGated7 rule authority.

    MRO (probe, SlotHybridPolicy, OccGated7Policy, ...): the hybrid's
    _chain_action (slot heads overwrite continuous commands) runs on
    top of the occ7 chain; modes/run bits come from occ7.
    """

    WEIGHTS = None

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec,
                         weights_path=self.WEIGHTS)


class SlotT1SoftProbe(_SlotHybrid7Mixin, SlotHybridPolicy,
                      OccGated7Policy):
    """MoE T3: occ7 authority + soft-k3 slot heads (T1 winner)."""

    KEY = "probe_slot_soft7"
    WEIGHTS = "smart_buildings/results/slot_runs/slot_t1_soft3.npz"


class SlotT1HardProbe(_SlotHybrid7Mixin, SlotHybridPolicy,
                      OccGated7Policy):
    """MoE T3: occ7 authority + hard-fit slot heads (deploy-blended)."""

    KEY = "probe_slot_hard7"
    WEIGHTS = "smart_buildings/results/slot_runs/slot_t1_hard.npz"


class MedSat05Probe(OccEconDbhHot9Policy):
    """D1: occ9 + median-demand SAT (their DEMAND_QUANTILE=0.5)."""

    KEY = "probe_medsat05"
    DEMAND_QUANTILE = 0.5


class UShift06Probe(OccEconDbhHot9Policy):
    """D2: occ9 + per-zone u-EMA economic target shift (0.6K, EMA144)."""

    KEY = "probe_ushift06"
    ZONE_SHIFT_K = 0.6
    ZONE_EMA_STEPS = 144
    ZONE_SHIFT_FULL_U = 0.2

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._u_ema = {}

    def _zone_target(self, obs, lo, hi):
        target = super()._zone_target(obs, lo, hi)
        if not self.ZONE_SHIFT_K:
            return target
        ema = self._u_ema.get(getattr(self, "_cur_vav", None))
        if ema is None:
            return target
        frac = float(np.clip(ema / self.ZONE_SHIFT_FULL_U, -1.0, 1.0))
        shift = min(self.ZONE_SHIFT_K, 0.4 * (hi - lo)) * frac
        return float(np.clip(target + shift, lo + 0.3, hi - 0.3))

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if self.ZONE_SHIFT_K > 0:
            for v, u in self._zone_u.items():
                prev = self._u_ema.get(v, u)
                self._u_ema[v] = prev + (u - prev) / self.ZONE_EMA_STEPS
        return action


class Wknd6Probe(OccEconDbhHot9Policy):
    """D3: occ9 + weekend heating setback (lo edge -6K)."""

    KEY = "probe_wknd6"
    WEEKEND_EXTRA_DROP_K = 6.0

    def _weekend_drop(self, obs):
        if not self.WEEKEND_EXTRA_DROP_K or self._pi_active:
            return 0.0
        _, day = self._get_hour_and_day(obs)
        return self.WEEKEND_EXTRA_DROP_K if day >= 5 else 0.0

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        return lo - self._weekend_drop(obs), hi

    def _ahu_error(self, obs, ctrl, low, high):
        return super()._ahu_error(obs, ctrl, low - self._weekend_drop(obs),
                                  high)


class ErrResProbe(OccEconDbhHot9Policy):
    """D4: occ9 + per-zone error-reserve band tightening (facilities v42).

    Zones with large tracking error get a symmetric reserve: the band
    narrows by `extra` on both edges, so the PI loop holds drifting
    zones tighter. errEMA tracks |temp - target| (clipped 2.0K) at
    0.94/0.06 from init 0.08; extra = min(0.5*(ema-0.08) in [0,0.25],
    0.25*bandwidth); comfort (PI-active) steps only. No _ahu_error
    override: the reserve is per-VAV and has no AHU-level equivalent
    (AHU ctrl sees the base band). Fullrun-only like D1-D3 (never a
    skynet rollout base), so the EMA updates exactly once per zone
    per step in every real path.
    """

    KEY = "probe_errres025"
    ERRRES_INIT = 0.08
    ERRRES_FAST = 0.06
    ERRRES_GAIN = 0.5
    ERRRES_CAP = 0.25
    ERRRES_BANDFRAC = 0.25
    ERRRES_CLIP_K = 2.0

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._err_ema = {}

    def _zone_err(self, obs, temp, lo, hi):
        err = super()._zone_err(obs, temp, lo, hi)
        vav = self._cur_vav
        if vav is not None:
            tgt = self._zone_target(obs, lo, hi)
            e = min(abs(float(temp) - float(tgt)), self.ERRRES_CLIP_K)
            ema = self._err_ema.get(vav, self.ERRRES_INIT)
            self._err_ema[vav] = ((1.0 - self.ERRRES_FAST) * ema
                                  + self.ERRRES_FAST * e)
        return err

    def _reserve(self, vav_id, lo, hi):
        if not self._pi_active:
            return 0.0
        ema = self._err_ema.get(vav_id, self.ERRRES_INIT)
        raw = self.ERRRES_GAIN * (ema - self.ERRRES_INIT)
        raw = min(max(raw, 0.0), self.ERRRES_CAP)
        return min(raw, self.ERRRES_BANDFRAC * max(hi - lo, 0.0))

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        extra = self._reserve(vav_id, lo, hi)
        return lo + extra, hi - extra
