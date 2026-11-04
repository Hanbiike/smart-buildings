# -*- coding: utf-8 -*-
"""R1 occupancy-conditional OFF levers on occ9 (plan-20 P20-5).

Two separately-gated sub-levers (screen each, promote each on its
own gate):

- OccOffA1Policy (KEY occ_offa1): night occupancy-spike override.
  The base `_decide` trusts instantaneous `_occupied`: one person at
  night (security/cleaning) holds the whole building in `normal`
  until morning. R1a re-reads occupied as empty when the 24h
  occupied-fraction EMA is below threshold AND outside comfort
  hours AND outside the prestart window (morning arrivals can
  never be overridden: prestart-relevant steps are exempt).
- OccOffB1Policy (KEY occ_offb1): water-integral reset on
  shutdown entry. `_enter_shutdown` resets zone integrals and AHU
  state but leaves `_water_integral` wound up overnight, so the
  first morning shortfall gets an overstated HWS fraction.
"""
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)


class OccOffA1Policy(OccEconDbhHot9Policy):
    """R1a: night occupancy-spike override."""

    KEY = "occ_offa1"
    OCC_24H_ALPHA = 1.0 - 1.0 / 288.0
    # Occupied-fraction EMA below this reads as "empty day/night".
    # A fully occupied 8h workday peaks near 8/24 ~= 0.33; nights
    # after occupied days decay below 0.3 well before midnight.
    SPIKE_EMA_MAX = 0.30

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._occ_ema_b = 0.0

    def _per_step(self, obs, mean_now):
        super()._per_step(obs, mean_now)
        occ = 1.0 if self._occupied(obs) else 0.0
        self._occ_ema_b = (self.OCC_24H_ALPHA * self._occ_ema_b
                           + (1.0 - self.OCC_24H_ALPHA) * occ)

    def _decide(self, obs, occupied, mean_now):
        if (occupied and self._occ_ema_b < self.SPIKE_EMA_MAX
                and not self._comfort_now(obs)):
            steps_left = self._steps_to_comfort(obs)
            if (steps_left is None
                    or steps_left > self._needed_prestart(mean_now)):
                occupied = False  # night spike: read as empty
        return super()._decide(obs, occupied, mean_now)


class OccOffB1Policy(OccEconDbhHot9Policy):
    """R1b: reset the HWS water integral on shutdown entry."""

    KEY = "occ_offb1"

    def _enter_shutdown(self):
        super()._enter_shutdown()
        if hasattr(self, "_water_integral"):
            self._water_integral = 0.0
