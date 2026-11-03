"""Heating-gated pre-start ramp on occ9 (plan 18/J1, rule-only).

Same RAMP_K weekday pre-start target ramp as `occ_ramp` (plan F/C3),
but applied ONLY when the plant heats: outdoor air below the comfort
band center. The C3 blanket ramp won heating climates (atl -0.26%,
den -1.12%, fb -0.52%) and lost cooling ones (phx +3.31%, sg +2.48%,
worst b1/phx5 +7.67%) — the cooling-side damage was the whole
regression. This gate keeps the wins and skips the losses.

Untested on occ9: their ramp-only-where-heats counter-evidence ran on
a base without occ shutdown (research 16:174-176).

Proceeds only if it beats `occ_econdbhhot9` on train-30 mean AND
worst climate.
"""

from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)
from smart_buildings.baselines.zone_pi import _outside_kelvin


class OccRampGated9Policy(OccEconDbhHot9Policy):
    """occ9 + pre-start ramp gated on heating regime (OAT < band)."""

    KEY = "occ_rampgated9"

    RAMP_K = 1.0
    RAMP_FLOOR_K = 289.0

    def _zone_target(self, obs, lo, hi):
        target = super()._zone_target(obs, lo, hi)
        if not self.RAMP_K:
            return target
        hour, day = self._get_hour_and_day(obs)
        if day >= 5 or hour >= self.morning_start_hour:
            return target
        if not self._heating_regime(obs, lo, hi):
            return target
        return max(self.RAMP_FLOOR_K,
                   target - self.RAMP_K * (self.morning_start_hour - hour))

    def _heating_regime(self, obs, lo, hi):
        """True when OAT sits below the band center (plant heats)."""
        try:
            if getattr(self, "_oat_any", None) is None:
                return True
            oat = _outside_kelvin(self, float(obs[self._oat_any]))
        except (TypeError, ValueError):
            return True
        return oat < 0.5 * (lo + hi)
