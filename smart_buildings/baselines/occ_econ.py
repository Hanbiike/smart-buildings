# -*- coding: utf-8 -*-
"""Explicit PI-hours economizer, distilled from sb-t-combo-d.ipynb (public 0.012).

Their DSA_HOP_Econ: forced-on AHUs keep recirculation at maximum (the
organizer economizer only runs inside the PID branch), so the chiller does
all the cooling even when outside air is colder than return air. In this
simulator the recirculation command does not change airflow, so outside air
costs no fan energy: draw enough to bring the mix to the supply-air target.
Heating-mode AHUs (SAT command >= 0) keep maximum recirculation.

Faithful port of their _base_action block (SAT native range 285-305 K).
Knobs: $ECON_ON (default 1; 0 = occ7-exact).
"""

import numpy as np

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin

_SAT_MIN_K, _SAT_MAX_K = 285.0, 305.0


class OccEconMixin:
    """Explicit economizer recirculation for cooling AHUs in PI hours."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._econ_on = int(_os.environ.get("ECON_ON", "1"))

    @staticmethod
    def _native(cmd, lo, hi):
        return lo + (float(cmd) + 1.0) * 0.5 * (hi - lo)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._econ_on or not self._pi_active:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            r_idx, oat_idx = ctrl.get("recirc_idx"), ctrl.get("oat_idx")
            if r_idx is None or oat_idx is None or not disc[ctrl["run_discrete_idx"]]:
                continue
            if float(cont[ctrl["temp_idx"]]) >= 0.0:
                continue
            target = self._native(cont[ctrl["temp_idx"]], _SAT_MIN_K, _SAT_MAX_K)
            try:
                inside = float(np.mean([
                    self._denormalize_zone_temp(obs[j])
                    for j in ctrl["zone_obs_indices"]]))
                oat = _outside_kelvin(self, obs[oat_idx])
            except (TypeError, ValueError):
                continue
            if oat >= inside or inside <= target:
                continue
            f_out = float(np.clip((inside - target) / (inside - oat), 0.3, 1.0))
            cont[r_idx] = np.float32(2.0 * (1.0 - f_out) - 1.0)
        return action


class OccEcon7Policy(OccEconMixin, OccGated7Policy):
    """occ_gated7 with explicit PI-hours economizer (T_COMBO_D distillation)."""

    KEY = "occ_econ7"
