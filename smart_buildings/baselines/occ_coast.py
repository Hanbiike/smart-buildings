# -*- coding: utf-8 -*-
"""Evening coast (research item 5 stop-side): lower point target after CS_START.

Workday occupancy falls off a cliff after ~13:00 (b1: 68 at noon -> 17 at
17:00 -> 4 at 19:00), yet conditioning runs full-blast into the evening.
During coast hours the PI point target drops by $CS_DROP below the
would-be target (down-shift included): point regulation, no deadband
pathology. Morning readiness is untouched.

Knobs: $CS_ON (default 1; 0 = occ7-exact), $CS_START (default 17.0),
$CS_DROP (default 1.0K).

Note: overriding _zone_band does NOT work — _command_zones resets any
band wider than comfort back to comfort (pre-occupancy guard). The target
is the only effective override point.
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class CoastMixin:
    """Coast-hour point-target drop for ZonePIPolicy._command_zones."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._cs_on = int(_os.environ.get("CS_ON", "1"))
        self._cs_start = float(_os.environ.get("CS_START", "17.0"))
        self._cs_drop = float(_os.environ.get("CS_DROP", "1.0"))

    def _zone_target(self, obs, lo, hi):
        target = super()._zone_target(obs, lo, hi)
        if not self._cs_on:
            return target
        try:
            hour, day = self._get_hour_and_day(obs)
        except (TypeError, ValueError):
            return target
        if day < 5 and hour >= self._cs_start:
            return float(target) - self._cs_drop
        return target


class OccCoast7Policy(CoastMixin, OccGated7Policy):
    """occ_gated7 with evening point-target coast."""

    KEY = "occ_coast7"
