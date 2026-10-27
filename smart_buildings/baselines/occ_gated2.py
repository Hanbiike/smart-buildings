"""Occupancy-gated gen-3.1, round 2: water-side heating economy.

Adds to OccGatedPolicy (kept in a separate module so round-1 batch results
stay comparable):

  OccGated2Policy
    F. Cold-weather hot-water floor: the water reset collapses to the zone PI
       demand in near-band steady state (~0.3), so lukewarm water forces high
       airflow to deliver the same heat -- and high airflow feeds the 30%
       outside-air ventilation load, which is over half the gas in occupied
       Fairbanks hours. When heating in cold weather, the loop is held at an
       OAT-scheduled floor instead; the zone PI self-balances to lower
       dampers, delivering the same heat with less air.
    G. WaterPumpIdleOff in the MRO: pump head + discrete off when no zone
       wants heat (occupied no-heat hours the shutdown does not cover).
  OccSat2Policy: round-2 + the OccSat damper cap (0.5).
"""

import numpy as np

from smart_buildings.baselines.occ_gated import (
    OccGatedPolicy,
    OccSatPolicy,
)
from smart_buildings.baselines.zone_pi import (
    DsatG4WPumpIdleOffPolicy,
    _outside_kelvin,
)


class _Round2Mixin:
    """Round-2 get_action: round-1 modes + hot-water floor.

    Enters the gen-3.1 chain via ``super(OccGatedPolicy, self)`` so the mode
    decision runs exactly once (calling OccGated.get_action would re-decide
    and double-count the comfort grace). The pump-off logic rides the normal
    ``_base_action`` chain through DsatG4WPumpIdleOffPolicy.
    """

    # OAT (K) at which the floor reaches 0 / full loop temperature.
    WATER_FLOOR_OAT_HI = 273.15
    WATER_FLOOR_OAT_LO = 243.15
    # Heat-active gate: below this zone demand / water fraction the floor
    # does not apply (no point heating a loop nobody draws from).
    WATER_FLOOR_MIN_HEAT = 0.05
    WATER_FLOOR_MIN_FRAC = 0.10

    def _water_floor_frac(self, obs) -> float:
        if getattr(self, "_oat_any", None) is None:
            return 0.0
        try:
            oat = _outside_kelvin(self, float(obs[self._oat_any]))
        except (TypeError, ValueError):
            return 0.0
        span = self.WATER_FLOOR_OAT_HI - self.WATER_FLOOR_OAT_LO
        if span <= 0:
            return 0.0
        return float(np.clip(
            (self.WATER_FLOOR_OAT_HI - oat) / span, 0.0, 1.0))

    def _apply_water_floor(self, obs, action) -> None:
        if self.water_temp_idx is None:
            return
        cont = action["continuous_action"]
        frac = (float(cont[self.water_temp_idx]) + 1.0) * 0.5
        heat_active = (getattr(self, "_max_heat", 0.0) > self.WATER_FLOOR_MIN_HEAT
                       or frac > self.WATER_FLOOR_MIN_FRAC)
        if not heat_active:
            return
        floor = self._water_floor_frac(obs)
        if floor > frac:
            cont[self.water_temp_idx] = np.float32(2.0 * floor - 1.0)

    def _chain_action(self, obs):
        """The gen-3.1 (+pump-off) chain with the hot-water floor applied."""
        action = super(OccGatedPolicy, self).get_action(obs)
        self._apply_water_floor(obs, action)
        return action

    def get_action(self, obs):
        occupied = self._occupied(obs)
        mean_now = self._mean_zone_temp(obs)
        self._per_step(obs, mean_now)
        mode = self._decide(obs, occupied, mean_now)

        if mode == "shutdown":
            self._prestart_active = False
            self._floor_mode = None
            self._enter_shutdown()
            return self._shutdown_action()

        if mode in ("floor_heat", "floor_cool"):
            self._prestart_active = False
            self._floor_mode = "heat" if mode == "floor_heat" else "cool"
            self._shutdown = False
            floor_heat, ceil_cool = self._floor_window()
            saved_eco = self.eco_temp_window
            saved_ceiling = self.NIGHT_COOL_CEILING
            saved_sensor = self.USE_SENSOR_BAND
            saved_hold = self.HOLD_STEPS
            self.eco_temp_window = (floor_heat, ceil_cool)
            self.NIGHT_COOL_CEILING = False
            self.USE_SENSOR_BAND = False
            self.HOLD_STEPS = 1
            try:
                return self._chain_action(obs)
            finally:
                self.eco_temp_window = saved_eco
                self.NIGHT_COOL_CEILING = saved_ceiling
                self.USE_SENSOR_BAND = saved_sensor
                self.HOLD_STEPS = saved_hold

        self._prestart_active = (mode == "prestart")
        self._floor_mode = None
        self._shutdown = False
        return self._chain_action(obs)


class OccGated2Policy(_Round2Mixin, DsatG4WPumpIdleOffPolicy, OccGatedPolicy):
    """OccGated + idle pump-off + cold-weather hot-water floor."""

    KEY = "occ_gated2"


class OccSat2Policy(_Round2Mixin, DsatG4WPumpIdleOffPolicy, OccSatPolicy):
    """OccSat (damper cap 0.5) + idle pump-off + hot-water floor."""

    KEY = "occ_sat2"
