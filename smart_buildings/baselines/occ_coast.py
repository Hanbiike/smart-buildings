# -*- coding: utf-8 -*-
"""Evening coast (research item 5 stop-side): eco band after CS_START.

Workday occupancy falls off a cliff after ~13:00 (b1: 68 at noon -> 17 at
17:00 -> 4 at 19:00), yet conditioning runs full-blast into the evening.
During coast hours (default >= 17:00) zones use the ECO band instead of
comfort: drift toward eco on both sides, saving heat + chill while only
the few remaining occupants are at stake. Morning readiness is untouched.

Knobs: $CS_ON (default 1; 0 = occ7-exact), $CS_START (default 17.0).
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class CoastMixin:
    """Eco-band substitution during evening coast hours."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._cs_on = int(_os.environ.get("CS_ON", "1"))
        self._cs_start = float(_os.environ.get("CS_START", "17.0"))

    def _zone_band(self, obs, vav_id, low, high):
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        if not self._cs_on:
            return lo, hi
        try:
            hour, day = self._get_hour_and_day(obs)
        except (TypeError, ValueError):
            return lo, hi
        if day < 5 and hour >= self._cs_start:
            eco_lo, eco_hi = self.eco_temp_window
            return float(eco_lo), float(eco_hi)
        return lo, hi


class OccCoast7Policy(CoastMixin, OccGated7Policy):
    """occ_gated7 with evening eco-band coast."""

    KEY = "occ_coast7"
