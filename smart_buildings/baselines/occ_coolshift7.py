# -*- coding: utf-8 -*-
"""Cooling-side target shift ported onto occ_gated7 (research item 4 tail).

Warm-side shift while cooling-dominated (+CSH_SHIFT when OAT > centre,
capped): the cooling analogue of the heating down-shift. Original
occ_coolshift sits on round-4 base and is unregistered; this is the fair
port. Knobs: $CSH_ON (default 1; 0 = occ7-exact), $CSH_SHIFT (0.3K).
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class CoolShiftMixin:
    """Warm-side PI target shift for OccGated7Policy."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._csh_on = int(_os.environ.get("CSH_ON", "1"))
        self._csh_shift = float(_os.environ.get("CSH_SHIFT", "0.3"))

    def _zone_target(self, obs, lo, hi):
        base = super()._zone_target(obs, lo, hi)
        if not self._csh_on or self._oat_any is None:
            return base
        try:
            outside = _outside_kelvin(self, float(obs[self._oat_any]))
        except (TypeError, ValueError):
            return base
        centre = 0.5 * (lo + hi)
        if outside <= centre:
            return base
        shift = min(self._csh_shift, 0.15 * (hi - lo))
        return min(base + shift, hi - 0.1)


class OccCoolShift7Policy(CoolShiftMixin, OccGated7Policy):
    """occ_gated7 with cooling-side warm target shift."""

    KEY = "occ_coolshift7"
