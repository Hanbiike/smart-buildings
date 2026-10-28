# -*- coding: utf-8 -*-
"""G36 two-stage SAT reset on occ_gated7 (research item 6).

Stage 1 (slow anchor): OAT-scheduled SAT base, 18 C at OAT <= 16 C down to
12 C at OAT >= 21 C (G36 §5; normalized with the in-repo 10 K/unit slope:
-1.0 = 285.15 K, -0.4 = 291.15 K, matching COLD_SAT/WARM_SAT comments).
Stage 2 (fast trim): per-AHU Trim & Respond around the anchor on zone
cooling-request proxies (temp above band + damper open or PI u high):
0 requests -> +TRIM, 1 -> +RESP1, >=2 -> +RESP2 (G36 +0.1/-0.2/-0.6 K per
2 min, rescaled x2.5 to 5-min steps). Inert below SAT2_MIN_OAT (SAT action
dead there, plan §6); leaves super's single-stage demand SAT otherwise.

Knobs: $SAT2_ON (default 1; 0 = occ7-exact), $SAT2_TRIM=0.25,
$SAT2_RESP1=-0.5, $SAT2_RESP2=-1.5, $SAT2_REQ_DB=0.2, $SAT2_REQ_U=0.5.
"""

import numpy as np

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin

_K_PER_UNIT = 10.0
_NORM_ANCHOR_K = 295.15


def _sat_norm(kelvin: float) -> float:
    return float(np.clip((kelvin - _NORM_ANCHOR_K) / _K_PER_UNIT, -1.0, 1.0))


class OccGSatMixin:
    """G36 two-stage SAT reset for OccGated7Policy._base_action."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._sat2_on = int(_os.environ.get("SAT2_ON", "1"))
        self._sat2_trim = float(_os.environ.get("SAT2_TRIM", "0.25"))
        self._sat2_resp1 = float(_os.environ.get("SAT2_RESP1", "-0.5"))
        self._sat2_resp2 = float(_os.environ.get("SAT2_RESP2", "-1.5"))
        self._sat2_req_db = float(_os.environ.get("SAT2_REQ_DB", "0.2"))
        self._sat2_req_u = float(_os.environ.get("SAT2_REQ_U", "0.5"))
        self._sat2_min_oat = 266.35
        self._sat2_lo = float(_os.environ.get("SAT2_LO", "285.15"))
        self._sat2_hi = float(_os.environ.get("SAT2_HI", "291.15"))
        self._sat2_trim_state: dict[int, float] = {}

    def _sat2_base(self, oat_k: float) -> float:
        if oat_k <= 289.15:
            return self._sat2_hi
        if oat_k >= 294.15:
            return self._sat2_lo
        f = (oat_k - 289.15) / (294.15 - 289.15)
        return self._sat2_hi - f * (self._sat2_hi - self._sat2_lo)

    def _sat2_requests(self, obs, cont, ctrl) -> int:
        lo, hi = self.comfort_temp_window
        n = 0
        for vav_id, obs_idx, _ in ctrl["zones"]:
            try:
                t = self._denormalize_zone_temp(obs[obs_idx])
            except (TypeError, ValueError, IndexError):
                continue
            if t <= hi + self._sat2_req_db:
                continue
            u = float(self._zone_u.get(vav_id, 0.0))
            aidx = self.zone_action_indices.get(vav_id, {}).get(
                "supply_air_damper_percentage_command")
            damp = (float(cont[aidx]) + 1.0) * 0.5 if aidx is not None else 1.0
            if damp > 0.5 or u > self._sat2_req_u:
                n += 1
        return n

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._sat2_on or not self._pi_active:
            return action
        if self._oat_any is None:
            return action
        try:
            oat = _outside_kelvin(self, float(obs[self._oat_any]))
        except (TypeError, ValueError):
            return action
        if oat < self._sat2_min_oat:
            return action
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            if not disc[ctrl["run_discrete_idx"]]:
                continue
            base = self._sat2_base(oat)
            tr = self._sat2_trim_state.get(i, 0.0)
            n = self._sat2_requests(obs, cont, ctrl)
            if n == 0:
                tr += self._sat2_trim
            elif n == 1:
                tr += self._sat2_resp1
            else:
                tr += self._sat2_resp2
            tr = float(np.clip(tr, -6.0, 6.0))
            self._sat2_trim_state[i] = tr
            cont[ctrl["temp_idx"]] = np.float32(
                _sat_norm(float(np.clip(base + tr, self._sat2_lo,
                                        self._sat2_hi))))
        return action


class OccGSat7Policy(OccGSatMixin, OccGated7Policy):
    """occ_gated7 with G36 two-stage SAT reset."""

    KEY = "occ_gsat7"
