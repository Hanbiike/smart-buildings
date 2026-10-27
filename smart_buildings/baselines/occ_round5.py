"""Round-5 experimental variants on OccGated4 (all need full-episode proof).

  OccDutyMicroPolicy: duty cycling with a short coast (6 on / 1 off). The
    4/2 and 4/4 variants cost more comfort ripple than they saved in
    ventilation; a single 5-min coast keeps the drift (and the ON-sprint)
    small.
  OccCoolShiftPolicy: ride 0.3 K toward the warm edge while cooling-dominated
    (hot outside air). The symmetric 1.0 K shift was rejected for comfort;
    this is cooling-only and small.
  OccFloor287Policy: overnight floor cap at eco_low - 2 (= gen31's hold)
    instead of eco_low - 1, betting the gas saved beats the Monday regret.
"""

import numpy as np

from smart_buildings.baselines.occ_duty import OccDutyPolicy
from smart_buildings.baselines.occ_gated4 import OccGated4Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class OccDutyMicroPolicy(OccDutyPolicy):
    """Micro duty: 30 min on / 5 min off."""

    KEY = "occ_duty_micro"
    DUTY_ON_STEPS = 6
    DUTY_OFF_STEPS = 1


class OccCoolShiftPolicy(OccGated4Policy):
    """OccGated4 + small warm-side target shift while cooling-dominated."""

    KEY = "occ_coolshift"
    COOL_SHIFT_K = 0.3

    def _zone_target(self, obs, lo, hi):
        base = super()._zone_target(obs, lo, hi)
        if self._oat_any is None:
            return base
        try:
            outside = _outside_kelvin(self, float(obs[self._oat_any]))
        except (TypeError, ValueError):
            return base
        centre = 0.5 * (lo + hi)
        if outside <= centre:
            return base
        shift = min(self.COOL_SHIFT_K, 0.15 * (hi - lo))
        return min(base + shift, hi - 0.1)

    def _pi_hours(self, obs):
        return super()._pi_hours(obs)


class OccFloor287Policy(OccGated4Policy):
    """OccGated4 with the overnight floor capped one kelvin lower."""

    KEY = "occ_floor287"
    FLOOR_HEAT_CAP_BELOW_ECO = 2.0
