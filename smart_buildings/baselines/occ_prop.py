# -*- coding: utf-8 -*-
"""Occupancy-proportional conditioning (research item 7).

Comfort value scales with occupants present, energy cost does not. Scale
the PI error only when empty AND far from the threatened edge:
scale = 1-(1-OP_MIN)*(1-frac)*ride, frac = raw count / running episode max
(guarded: max < OP_MAXN -> frac = 1), ride = m below band centre (heating:
needs warmth) or 1-m above it (cooling: needs coolness), m=(temp-lo)/(hi-lo).
Cold buildings, hot buildings, peak hours always run full effort.

Knobs: $OP_ON (default 1; 0 = occ7-exact), $OP_MIN (default 0.5, effort
floor at zero occupancy), $OP_MAXN (default 5).
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class OccPropMixin:
    """Occupancy-fraction error scaling for ZonePIPolicy._command_zones."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._op_on = int(_os.environ.get("OP_ON", "1"))
        self._op_min = float(_os.environ.get("OP_MIN", "0.5"))
        self._op_maxn = float(_os.environ.get("OP_MAXN", "5"))
        self._occ_max = 0.0

    def _zone_err(self, obs, temp, lo, hi):
        err = super()._zone_err(obs, temp, lo, hi)
        if not self._op_on or self._occ_idx is None:
            return err
        try:
            n = float(obs[self._occ_idx])
        except (TypeError, ValueError):
            return err
        if n > self._occ_max:
            self._occ_max = n
        if self._occ_max < self._op_maxn:
            return err
        frac = n / self._occ_max
        span = hi - lo
        m = 0.0 if span <= 0 else min(1.0, max(0.0, (temp - lo) / span))
        # Coast only when the building can ride to next occupancy: empty
        # AND far from the threatened edge. Heating need (temp<=centre):
        # ride needs warmth (m~1); cooling need: ride needs coolness
        # (m~0). Cold buildings, hot buildings, and peak hours (frac~1)
        # always run full effort (fb1/phx5 lessons).
        centre = 0.5 * (lo + hi)
        ride = m if temp <= centre else 1.0 - m
        scale = 1.0 - (1.0 - self._op_min) * (1.0 - frac) * ride
        return float(err) * scale


class OccProp7Policy(OccPropMixin, OccGated7Policy):
    """occ_gated7 with occupancy-proportional PI effort."""

    KEY = "occ_prop7"
