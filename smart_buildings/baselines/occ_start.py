# -*- coding: utf-8 -*-
"""Adaptive optimal start, research item 5 (handoff from item 3).

Item 3 closed negative: blanket margin/safety cuts crash cold mornings
(Denver +9.2%) while flat elsewhere — margin is load-bearing where it is
cold and waste where it is mild. So schedule the margin on pre-dawn OAT:

  margin_steps = PRESTART_MARGIN_STEPS + ST_OAT_K * (ST_OAT_REF - OAT)

Cold pre-dawn -> extra margin (protect recovery); mild pre-dawn -> margin
cut (save energy). The gap/rate x SAFETY core is untouched JCI equation.

Knobs: $ST_ON (default 1; 0 = occ7-exact), $ST_OAT_K (default 0.0 steps/K;
0 = occ7-exact), $ST_OAT_REF (default 283.15 K).
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class OccStartMixin:
    """OAT-scheduled prestart margin for the _needed_prestart equation."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._st_on = int(_os.environ.get("ST_ON", "1"))
        self._st_oat_k = float(_os.environ.get("ST_OAT_K", "0.0"))
        self._st_oat_ref = float(_os.environ.get("ST_OAT_REF", "283.15"))
        self._st_obs = None

    def _per_step(self, obs, mean_now):
        self._st_obs = obs
        return super()._per_step(obs, mean_now)

    def _needed_prestart(self, mean_now: float) -> int:
        base = super()._needed_prestart(mean_now)
        if (not self._st_on or not self._st_oat_k or self._oat_any is None
                or self._st_obs is None):
            return base
        try:
            oat = _outside_kelvin(self, float(self._st_obs[self._oat_any]))
        except (TypeError, ValueError, IndexError):
            return base
        extra = int(round(self._st_oat_k * (self._st_oat_ref - oat)))
        return max(0, base + extra)


class OccStart7Policy(OccStartMixin, OccGated7Policy):
    """occ_gated7 with OAT-scheduled adaptive prestart margin."""

    KEY = "occ_start7"
