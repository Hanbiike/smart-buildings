# -*- coding: utf-8 -*-
"""Physics-based heating damper cap, distilled from sb-t-combo-d.ipynb (0.012).

Their DSA_HeatOptP: cap each heating zone's damper at the airflow that
maximises delivered heat. Reheat coil: discharge = T_s + e (T_w - T_s),
e = min(1, C_w/C_a, (35 C - T_s)/(T_w - T_s)) — verified against the local
sim source (smart_control/simulator/vav.py). Heat into the zone rises with
air while the 35 C discharge cap binds, then falls once the water side
limits (extra air is mixed air the coil can no longer warm). The peak is at
C_a* = C_w (T_w - T_s)/(35 C - T_s). Each coil's water flow is
Kv*sqrt(dP)/3600 from live observations (water temp, pump dP, OAT) and this
step's own commands (valve, recirculation, SAT, pressure). With pressure
opt: raise static pressure so saturated zones (u <= SAT_HEAT_U) reach their
optimal airflow.

Knobs: $HOPT_ON (default 1; 0 = occ7-exact), $HOPT_POPT (default 1; 0 =
damper cap only, no pressure raise), $HOPT_SAT_U (default -0.9).
"""

import numpy as np

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class HeatOptMixin:
    """Coil-physics damper cap + saturation pressure opt."""

    HOPT_VAV_MAX_FLOW_M3S = 2.0
    HOPT_VAV_DESIGN_PRESSURE_PA = 2000.0
    HOPT_REHEAT_KV_MAX = 0.108
    HOPT_MAX_DISCHARGE_K = 308.15
    HOPT_AIR_CP_RHO = 1.2 * 1006.0
    HOPT_WATER_CP_RHO = 1000.0 * 4180.0
    HOPT_SAT_MIN_K = 285.0
    HOPT_SAT_MAX_K = 305.0

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._hopt_on = int(_os.environ.get("HOPT_ON", "1"))
        self._hopt_popt = int(_os.environ.get("HOPT_POPT", "1"))
        self._hopt_sat_u = float(_os.environ.get("HOPT_SAT_U", "-0.9"))
        names = list(self.field_names)
        self._tw_idx = (names.index("hws_supply_water_temperature_sensor")
                        if "hws_supply_water_temperature_sensor" in names
                        else None)
        self._dp_idx = (names.index("hws_differential_pressure_sensor")
                        if "hws_differential_pressure_sensor" in names
                        else None)

    @staticmethod
    def _native(cmd, lo, hi):
        return lo + (float(cmd) + 1.0) * 0.5 * (hi - lo)

    def _coil_targets(self, obs, cont, ctrl, pressure_pa):
        tw = float(obs[self._tw_idx])
        dp = max(float(obs[self._dp_idx]), 0.0)
        oat_idx = ctrl.get("oat_idx")
        if oat_idx is None or tw <= self.HOPT_MAX_DISCHARGE_K or dp <= 0.0:
            return []
        oat = _outside_kelvin(self, obs[oat_idx])
        t_ret = float(np.mean([
            self._denormalize_zone_temp(obs[j])
            for j in ctrl["zone_obs_indices"]]))
        recirc = 0.7
        if ctrl.get("recirc_idx") is not None:
            recirc = min(0.7, max(0.0, self._native(
                cont[ctrl["recirc_idx"]], 0.0, 1.0)))
        t_mix = recirc * t_ret + (1.0 - recirc) * oat
        sat = self._native(cont[ctrl["temp_idx"]],
                           self.HOPT_SAT_MIN_K, self.HOPT_SAT_MAX_K)
        t_s = sat if t_mix > sat else t_mix
        if t_s >= self.HOPT_MAX_DISCHARGE_K:
            return []
        out = []
        unit_flow = (self.HOPT_VAV_MAX_FLOW_M3S
                     * np.sqrt(pressure_pa / self.HOPT_VAV_DESIGN_PRESSURE_PA))
        for vav_id, _, acts in ctrl["zones"]:
            u = self._zone_u.get(vav_id)
            d_idx = acts.get("supply_air_damper_percentage_command")
            r_idx = acts.get("reheat_valve_percentage_command")
            if u is None or u >= 0.0 or d_idx is None or r_idx is None:
                continue
            valve = (float(cont[r_idx]) + 1.0) * 0.5
            c_w = (self.HOPT_WATER_CP_RHO * valve * self.HOPT_REHEAT_KV_MAX
                   * np.sqrt(dp) / 3600.0)
            flow = (c_w * (tw - t_s) / (self.HOPT_MAX_DISCHARGE_K - t_s)
                    / self.HOPT_AIR_CP_RHO)
            out.append((d_idx, flow / unit_flow, flow, u))
        return out

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._hopt_on or not self._pi_active:
            return action
        if self._tw_idx is None or self._dp_idx is None:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            if not disc[ctrl["run_discrete_idx"]] or ctrl["pressure_idx"] is None:
                continue
            p_pa = max(200.0, self._native(
                cont[ctrl["pressure_idx"]], 0.0, 2000.0))
            targets = self._coil_targets(obs, cont, ctrl, p_pa)
            if not targets:
                continue
            if self._hopt_popt:
                us = [self._zone_u[v] for v, _, _ in ctrl["zones"]
                      if v in self._zone_u]
                needed = [f for _, _, f, u in targets if u <= self._hopt_sat_u]
                if us and float(np.mean(us)) < 0.0 and needed:
                    p_opt = (self.HOPT_VAV_DESIGN_PRESSURE_PA
                             * (max(needed) / self.HOPT_VAV_MAX_FLOW_M3S) ** 2)
                    p_new = float(np.clip(p_opt, 200.0, 2000.0))
                    if p_new > p_pa:
                        p_pa = p_new
                        cont[ctrl["pressure_idx"]] = np.float32(p_pa / 1000.0 - 1.0)
                        targets = self._coil_targets(obs, cont, ctrl, p_pa)
            for d_idx, d_opt, _, _ in targets:
                d_now = (float(cont[d_idx]) + 1.0) * 0.5
                d_new = min(d_now, max(self.MIN_DAMPER, d_opt))
                cont[d_idx] = np.float32(2.0 * d_new - 1.0)
        return action


class OccHeatOpt7Policy(HeatOptMixin, OccGated7Policy):
    """occ_gated7 with coil-physics heating damper cap (T_COMBO_D distillation)."""

    KEY = "occ_heatopt7"
