"""Occupancy-gated gen-3.1, round 7: occupied-only trim-and-respond.

Round-6 (occ_gated6) added OAT-aware trim-and-respond static pressure and cut
big-zone comfort bleed (transfer -13%), but prestart recovery ramps in deep
cold look like starvation (cold zones, open dampers) and misfire the trigger
(train_2 February +8%). True bleeds persist into occupied hours, so round-7
responds only when occupied: prestart and floor keep minimum pressure.

Validated over 6 tail/misfire pairs: xfer_3 heat tail -23%, xfer_3 cooling
tail -34%, all misfire pairs flat to -1% (train_1/train_2 February bit-equal
to occ_gated4).
"""

import numpy as np

from smart_buildings.baselines.occ_gated4 import OccGated4Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class OccGated7Policy(OccGated4Policy):
    """Round-6 trim-and-respond, occupied-only trigger (no prestart)."""

    KEY = "occ_gated7"

    PTR_UP = 0.25
    PTR_DOWN = 0.03
    PTR_MAX = 0.0
    PTR_DSAT = 0.85
    PTR_DB = 1.0
    PTR_DB_SLOPE = 0.1
    PTR_DB_REF = 10.0
    PTR_PERSIST = 4

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._ptr_p = [-1.0] * len(self.ahu_controllers)
        self._ptr_nstarve = [0] * len(self.ahu_controllers)

    def _base_action(self, obs):
        action = super()._base_action(obs)
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
                self._ptr_p[i] = min(self.PTR_MAX, self._ptr_p[i] + self.PTR_UP)
            elif all_closed:
                self._ptr_p[i] = max(-1.0, self._ptr_p[i] - self.PTR_DOWN)
            cont[pidx] = np.float32(self._ptr_p[i])
        return action
