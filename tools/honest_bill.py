"""Honest boiler billing for local measurement only (plan F/B1, never shipped).

Ports `/tmp/rl_zip/sbrl.py::honest_bill` onto our vendored boiler
(`smart_buildings/smart_control/simulator/boiler.py`). The shipped
`compute_thermal_energy_rate` bills the hot-water flow at the tank
SETPOINT while the coils draw at the tank ACTUAL temperature, so heat
drawn with setpoint below tank temperature is never paid for. The
patch bills the flow at `max(tank T, return T)` plus dissipation at
that temperature plus the tank-storage term. Dynamics are untouched:
only the gas number the reward reads changes.

Idempotent; kept out of `smart_buildings/baselines/` per the plan
(group-B definition of done). Apply in a driver process before
building the environment:

    from tools.honest_bill import apply_honest_bill
    apply_honest_bill()
"""

from smart_buildings.smart_control.simulator import boiler as _boiler
from smart_buildings.smart_control.utils import constants as _constants


def apply_honest_bill() -> None:
    """Replace the billing rate with the tank-temperature bill, once."""
    B = _boiler.Boiler
    if getattr(B, "_honest_bill", False):
        return

    def compute_thermal_energy_rate(self, return_water_temp,
                                    outside_temp, total_flow_rate):
        # Stored tank state; `_set_current_temperature` advances it.
        supply_water_temp = max(self._current_temperature,
                                return_water_temp)
        flow = (_constants.WATER_HEAT_CAPACITY
                * _constants.WATER_DENSITY
                * total_flow_rate
                * (supply_water_temp - return_water_temp))
        dissipation = self.compute_thermal_dissipation_rate(
            supply_water_temp, outside_temp)
        secs = self._last_step_duration.total_seconds()
        tank = (_constants.WATER_HEAT_CAPACITY
                * _constants.WATER_DENSITY
                * self._water_capacity
                * self._step_tank_temperature_change / secs) if secs > 0 else 0
        return max(0.0, flow + dissipation + tank)

    B.compute_thermal_energy_rate = compute_thermal_energy_rate
    B._honest_bill = True
