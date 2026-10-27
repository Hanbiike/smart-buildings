"""Occupancy-gated gen-3.1, round 3: latch the prestart, bound the floor.

Round-1 full episodes showed two defects (kept separate so the round-1 batch
stays comparable):

  1. Prestart flicker: the need-vs-time-to-comfort threshold has no
     hysteresis, so the AHUs chatter on/off through the prestart as the mean
     zone temperature crosses the boundary -- including a shutdown blip at
     06:00 with occupants arriving. Fix: once prestart triggers it latches
     until comfort hours or occupancy.
  2. Recoverability floor too hot in saturation: in deep cold the plant
     cannot reach the computed floor, so floor_heat burns maximum gas all
     night for a Monday arrival that is cold anyway. Fix: bound the floor
     window at eco_low - 1 / eco_high + 1 (never holds warmer than a
     schedule-anchored night band), with 1 K hysteresis on entry/exit.

Keeps round-2 (idle pump-off, hot-water floor) underneath.
"""

from smart_buildings.baselines.occ_gated2 import (
    OccGated2Policy,
    OccSat2Policy,
)


class _Round3Mixin:
    """Latched prestart + bounded hysteretic floor."""

    FLOOR_ENTER_DB_K = 0.5
    FLOOR_EXIT_DB_K = 0.5
    FLOOR_HEAT_CAP_BELOW_ECO = 1.0   # never hold warmer than eco_low - 1
    FLOOR_COOL_CAP_ABOVE_ECO = 1.0   # never hold cooler than eco_high + 1

    def _floor_window(self) -> tuple[float, float]:
        floor_heat, ceil_cool = super()._floor_window()
        eco_lo, eco_hi = self.eco_temp_window
        floor_heat = min(floor_heat, eco_lo - self.FLOOR_HEAT_CAP_BELOW_ECO)
        ceil_cool = min(ceil_cool, eco_hi + self.FLOOR_COOL_CAP_ABOVE_ECO)
        return floor_heat, ceil_cool

    def _update_rates(self, mean_now: float) -> None:
        # Learn recovery rates from prestart only. Occupied steady-state
        # micro-rises (strong heat demand holding against losses with a near-
        # zero delta) drag the EMA toward zero, which pushes the overnight
        # floor up to the band edge within days. Prestart is unoccupied and
        # plant-driven, so its deltas are the rate that actually sizes the
        # next prestart. _last_mean still tracks every step so the first
        # prestart delta is genuine.
        if not self._prestart_active:
            self._last_mean = mean_now
            return
        super()._update_rates(mean_now)

    def _decide(self, obs, occupied: bool, mean_now: float) -> str:
        comfort_now = self._comfort_now(obs)
        if comfort_now:
            self._comfort_steps += 1
        else:
            self._comfort_steps = 0
            self._ever_occupied_today = False
        if occupied:
            self._ever_occupied_today = True

        if occupied:
            self._prestart_latched = False
            return "normal"
        if comfort_now:
            self._prestart_latched = False
            if self._ever_occupied_today:
                return "shutdown"
            if self._comfort_steps <= self.EMPTY_GRACE_STEPS:
                return "normal"  # arrivals pending
            return "shutdown"

        # Empty and outside comfort hours. A triggered prestart latches until
        # arrival: re-evaluating the need every step chatters the plant as
        # the mean crosses the threshold.
        if self._prestart_latched:
            return "prestart"
        steps_left = self._steps_to_comfort(obs)
        if steps_left is not None and steps_left <= self._needed_prestart(mean_now):
            self._prestart_latched = True
            return "prestart"

        floor_heat, ceil_cool = self._floor_window()
        if self._floor_mode == "heat":
            if mean_now > floor_heat + self.FLOOR_EXIT_DB_K:
                return "shutdown"
            return "floor_heat"
        if self._floor_mode == "cool":
            if mean_now < ceil_cool - self.FLOOR_EXIT_DB_K:
                return "shutdown"
            return "floor_cool"
        if mean_now < floor_heat - self.FLOOR_ENTER_DB_K:
            return "floor_heat"
        if mean_now > ceil_cool + self.FLOOR_ENTER_DB_K:
            return "floor_cool"
        return "shutdown"


class OccGated3Policy(_Round3Mixin, OccGated2Policy):
    """Round-3 assembly: latch + bounded floor on round-2."""

    KEY = "occ_gated3"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._prestart_latched = False


class OccSat3Policy(_Round3Mixin, OccSat2Policy):
    """Round-3 assembly on the saturation-restraint branch (experimental)."""

    KEY = "occ_sat3"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._prestart_latched = False
