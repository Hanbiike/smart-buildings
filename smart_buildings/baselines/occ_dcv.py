# -*- coding: utf-8 -*-
"""Pseudo-DCV recirculation schedule (research item 10b).

Ventilation-driven recirc, distinct from the economics-driven economizer:
when outside air is a load (OAT > return air), the outside-air fraction
follows building occupancy — a half-empty building needs half the
ventilation. f_out = VENT_MIN + (1 - VENT_MIN) * occ_frac, with occ_frac =
raw count / running episode max (guarded like item 7). When OAT <= return
air the economizer owns the recirc channel (free cooling) and this mixin
stays out. Standard DCV+economizer arbitration.

Knobs: $DCV_ON (default 1; 0 = occ7-exact), $DCV_VENT_MIN (default 0.3),
$DCV_MAXN (default 5).
"""

import numpy as np

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class OccDcvMixin:
    """Occupancy-scheduled recirculation when ventilation is a load."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._dcv_on = int(_os.environ.get("DCV_ON", "1"))
        self._dcv_vent_min = float(_os.environ.get("DCV_VENT_MIN", "0.3"))
        self._dcv_maxn = float(_os.environ.get("DCV_MAXN", "5"))
        self._dcv_occ_max = 0.0

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._dcv_on or not self._pi_active:
            return action
        if self._occ_idx is None or self._oat_any is None:
            return action
        try:
            n = float(obs[self._occ_idx])
            oat = _outside_kelvin(self, float(obs[self._oat_any]))
        except (TypeError, ValueError):
            return action
        if n > self._dcv_occ_max:
            self._dcv_occ_max = n
        if self._dcv_occ_max < self._dcv_maxn:
            return action
        frac = n / self._dcv_occ_max
        f_out = self._dcv_vent_min + (1.0 - self._dcv_vent_min) * frac
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            r_idx = ctrl.get("recirc_idx")
            if r_idx is None or not disc[ctrl["run_discrete_idx"]]:
                continue
            try:
                inside = float(np.mean([
                    self._denormalize_zone_temp(obs[j])
                    for j in ctrl["zone_obs_indices"]]))
            except (TypeError, ValueError):
                continue
            if oat <= inside:
                continue
            cur_out = 1.0 - (float(cont[r_idx]) + 1.0) * 0.5
            if f_out < cur_out:
                cont[r_idx] = np.float32(2.0 * (1.0 - f_out) - 1.0)
        return action


class OccDcv7Policy(OccDcvMixin, OccGated7Policy):
    """occ_gated7 with pseudo-DCV recirculation schedule."""

    KEY = "occ_dcv7"
