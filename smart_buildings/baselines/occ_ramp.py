"""Pre-start ramp port of the Skynet tuned rule (plan F/C3, rule-only).

On weekdays before `morning_start_hour`, each zone's PI target sits
`RAMP_K` per hour-to-morning below the base target, floored at
`RAMP_FLOOR_K` — a verbatim port of the `RAMP_K` branch in
`/tmp/rl_zip/zt_controller.py::_zone_target` (research 4.4) onto
`OccGated7Policy`. Nobody is in yet, so the lowered target prices no
regret; the target reaches the band exactly at the morning start.

Their reference gain (honest bill): b8 0.01109 -> 0.01077,
b7-rule-family 0.00749 -> 0.00734. This variant proceeds only if it
beats `occ_gated7` on train-30 mean AND worst climate.
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class OccRampPolicy(OccGated7Policy):
    """occ_gated7 + weekday pre-start target ramp (RAMP_K K/h, floored)."""

    KEY = "occ_ramp"

    RAMP_K = 1.0
    RAMP_FLOOR_K = 289.0

    def _zone_target(self, obs, lo, hi):
        target = super()._zone_target(obs, lo, hi)
        if self.RAMP_K:
            hour, day = self._get_hour_and_day(obs)
            if day < 5 and hour < self.morning_start_hour:
                target = max(self.RAMP_FLOOR_K,
                             target - self.RAMP_K * (self.morning_start_hour - hour))
        return target
