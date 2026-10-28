# -*- coding: utf-8 -*-
"""Per-AHU demand staging on occ_gated7 (research item 9).

Today PI hours force ALL AHUs on (partial states <1.5% on atlanta_3's
9 AHUs). Stage per AHU by local need with persistence: an AHU whose zones
are all inside the comfort band (+DB) for OFF_DELAY steps turns off; any
zone breaching the band turns it back on immediately. No staging during
prestart/floor/shutdown (recovery keeps full plant).

Knobs: $AHU9_ON (default 1; 0 = occ7-exact), $AHU9_OFF_DELAY (default 12
steps = 1 h), $AHU9_DB (default 0.2 K), $AHU9_U (default 0.1: demand-based
need on |u|, earlier than temp breach; 0 = temp-breach need).
"""

import numpy as np

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class OccStageMixin:
    """Per-AHU on/off staging for OccGated7Policy._base_action."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._ahu9_on = int(_os.environ.get("AHU9_ON", "1"))
        self._ahu9_off_delay = int(_os.environ.get("AHU9_OFF_DELAY", "12"))
        self._ahu9_db = float(_os.environ.get("AHU9_DB", "0.2"))
        self._ahu9_u = float(_os.environ.get("AHU9_U", "0.1"))
        self._ahu9_calm = [0] * len(self.ahu_controllers)

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._ahu9_on:
            return action
        if getattr(self, "_prestart_active", False):
            return action
        if getattr(self, "_floor_mode", None) is not None:
            return action
        if getattr(self, "_shutdown", False):
            return action
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        lo, hi = self.comfort_temp_window
        for i, ctrl in enumerate(self.ahu_controllers):
            if ctrl["run_discrete_idx"] is None:
                continue
            need = False
            for vav_id, obs_idx, _ in ctrl["zones"]:
                if self._ahu9_u > 0:
                    if abs(float(self._zone_u.get(vav_id, 0.0))) > self._ahu9_u:
                        need = True
                        break
                    continue
                try:
                    t = self._denormalize_zone_temp(obs[obs_idx])
                except (TypeError, ValueError, IndexError):
                    need = True
                    break
                if t < lo - self._ahu9_db or t > hi + self._ahu9_db:
                    need = True
                    break
            if need:
                self._ahu9_calm[i] = 0
                continue
            self._ahu9_calm[i] += 1
            if self._ahu9_calm[i] >= self._ahu9_off_delay:
                disc[ctrl["run_discrete_idx"]] = 0
        return action


class OccStage7Policy(OccStageMixin, OccGated7Policy):
    """occ_gated7 with per-AHU demand staging."""

    KEY = "occ_stage7"
