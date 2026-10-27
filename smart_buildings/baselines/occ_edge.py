# -*- coding: utf-8 -*-
"""Band-edge riding (research item 2): deadband PI error.

Comfort is free anywhere inside the band, but the PI holds the centre while
the median occupied regret is exactly 0. This mixin replaces the error with
a deadband: heat only below lo+backoff (target: the low edge), cool only
above hi-backoff (target: the high edge), coast inside. Hysteresis comes
from the deadband itself.

Knobs: $EB_ON (default 1; 0 = band-centre error, occ7-exact),
$EB_BACKOFF (default 0.25K safety margin from the edge), $EB_KP (default
empty = keep class KP).
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class EdgeRideMixin:
    """Deadband edge-riding error for ZonePIPolicy._command_zones."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._eb_on = int(_os.environ.get("EB_ON", "1"))
        self._eb_backoff = float(_os.environ.get("EB_BACKOFF", "0.25"))
        kp = _os.environ.get("EB_KP", "")
        if kp:
            self.KP = float(kp)

    def _zone_err(self, obs, temp, lo, hi):
        if not self._eb_on:
            return super()._zone_err(obs, temp, lo, hi)
        temp = float(temp)
        b = self._eb_backoff
        if temp < lo + b:
            return temp - (lo + b)
        if temp > hi - b:
            return temp - (hi - b)
        return 0.0


class OccEdge7Policy(EdgeRideMixin, OccGated7Policy):
    """occ_gated7 with band-edge-riding zone error."""

    KEY = "occ_edge7"
