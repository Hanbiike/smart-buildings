"""Focused tests for the local-only honest-bill patch (plan F/B1)."""

import unittest

import pandas as pd

from smart_buildings.smart_control.simulator import boiler as boiler_mod
from smart_buildings.smart_control.utils import constants
from tools import honest_bill

ORIG = boiler_mod.Boiler.compute_thermal_energy_rate


def _boiler(setpoint=320.0, tank=340.0, tank_change=0.0, secs=300.0):
    b = boiler_mod.Boiler(
        supply_water_temperature_setpoint=setpoint,
        heating_rate=1.0, cooling_rate=1.0)
    b._current_temperature = tank
    b._step_tank_temperature_change = tank_change
    b._last_step_duration = pd.Timedelta(secs, unit='second')
    return b


class HonestBillTest(unittest.TestCase):

    def tearDown(self):
        boiler_mod.Boiler.compute_thermal_energy_rate = ORIG
        boiler_mod.Boiler._honest_bill = False

    def test_shipped_bills_setpoint(self):
        # The bug B1 ports the fix for: setpoint 320 K, tank 340 K,
        # return 300 K -> shipped bills the flow at 320 K, and the
        # 20 K drawn from the hot tank go unbilled.
        b = _boiler()
        got = b.compute_thermal_energy_rate(300.0, 280.0, 0.01)
        flow = 4180.0 * 1000.0 * 0.01 * (320.0 - 300.0)
        diss = b.compute_thermal_dissipation_rate(320.0, 280.0)
        self.assertAlmostEqual(got, flow + diss, places=6)

    def test_patched_bills_tank(self):
        honest_bill.apply_honest_bill()
        b = _boiler()
        got = b.compute_thermal_energy_rate(300.0, 280.0, 0.01)
        flow = 4180.0 * 1000.0 * 0.01 * (340.0 - 300.0)
        diss = b.compute_thermal_dissipation_rate(340.0, 280.0)
        self.assertAlmostEqual(got, flow + diss, places=6)
        self.assertGreater(
            got, 4180.0 * 1000.0 * 0.01 * (320.0 - 300.0))

    def test_patched_tank_term_and_floor(self):
        honest_bill.apply_honest_bill()
        # Cooling tank releases more than flow+dissipation need
        # (tank -10.45 MW vs flow + dissipation ~2.1 MW).
        self.assertEqual(
            _boiler(tank_change=-500.0).compute_thermal_energy_rate(
                300.0, 280.0, 0.01), 0.0)
        # Warming tank adds its storage term.
        b = _boiler(tank_change=2.0)
        got = b.compute_thermal_energy_rate(300.0, 280.0, 0.01)
        tank = (constants.WATER_HEAT_CAPACITY * constants.WATER_DENSITY
                * b._water_capacity * 2.0 / 300.0)
        flow = 4180.0 * 1000.0 * 0.01 * (340.0 - 300.0)
        diss = b.compute_thermal_dissipation_rate(340.0, 280.0)
        self.assertAlmostEqual(got, flow + diss + tank, places=6)

    def test_return_above_tank_bills_zero_flow(self):
        honest_bill.apply_honest_bill()
        b = _boiler()
        got = b.compute_thermal_energy_rate(350.0, 280.0, 0.01)
        diss = b.compute_thermal_dissipation_rate(350.0, 280.0)
        self.assertAlmostEqual(got, diss, places=6)

    def test_dynamics_untouched(self):
        # Billing reads state without mutating it: tank temperature,
        # step change and setpoint are identical before/after.
        honest_bill.apply_honest_bill()
        b = _boiler()
        before = (b._current_temperature, b._step_tank_temperature_change,
                  b._supply_water_temperature_setpoint)
        b.compute_thermal_energy_rate(300.0, 280.0, 0.01)
        after = (b._current_temperature, b._step_tank_temperature_change,
                 b._supply_water_temperature_setpoint)
        self.assertEqual(before, after)

    def test_apply_idempotent(self):
        honest_bill.apply_honest_bill()
        first = boiler_mod.Boiler.compute_thermal_energy_rate
        honest_bill.apply_honest_bill()
        self.assertIs(boiler_mod.Boiler.compute_thermal_energy_rate, first)


if __name__ == '__main__':
    unittest.main(verbosity=2)
