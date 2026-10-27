"""Physical invariants; no simulator runs, fitted data or control changes."""

import unittest
import numpy as np

from smart_buildings.baselines.equipment_balance import (
    AIR_CAPACITY, WATER_CAPACITY, ZoneThermalParameters, air_flows, air_handler,
    water_flows, reheat_coil, zone_derivatives,
)


class EquipmentBalanceTest(unittest.TestCase):
    def test_airflow_pressure_cap_and_off(self):
        full = air_flows([1, 1], [2, 2], 2000, 2000, 3, running=True)
        np.testing.assert_allclose(full, [1.5, 1.5])
        half = air_flows([1], [2], 500, 2000, 3, running=True)
        np.testing.assert_allclose(half, [1])
        np.testing.assert_array_equal(air_flows([1], [2], 2000, 2000, 3, running=False), [0])

    def test_ahu_does_not_heat_mixed_air_to_setpoint(self):
        result = air_handler(294, 254, 0.7, 305, 1, 200, 0.8, running=True)
        self.assertEqual(result.supply_temperature, 282)
        self.assertEqual(result.cooling_power, 0)
        self.assertEqual(result.fan_power, 250)

    def test_ahu_cooling_balance(self):
        result = air_handler(304, 310, 1, 290, 2, 200, 1, running=True)
        self.assertEqual(result.cooling_power, AIR_CAPACITY * 2 * 14)

    def test_water_units_and_no_false_competition(self):
        a = water_flows([1, 1], [.108, .108], 20, 0, running=True)
        b = water_flows([1, 0], [.108, .108], 20, 0, running=True)
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(a[0], .108 * np.sqrt(20) / 3600)
        coupled = water_flows([1, 1], [.108, .108], 20, 1, running=True)
        self.assertLess(coupled[0], a[0])
        np.testing.assert_array_equal(water_flows([1], [.108], 20, 0, running=False), [0])

    def test_coil_conserves_both_air_and_water_energy(self):
        coil = reheat_coil(290, 340, .5, .0001, 308.15)
        self.assertAlmostEqual(coil.heat_to_air, AIR_CAPACITY * .5 * (coil.discharge_temperature - 290))
        self.assertAlmostEqual(coil.heat_to_air, WATER_CAPACITY * .0001 * (340 - coil.return_water_temperature))
        self.assertLessEqual(coil.discharge_temperature, 308.15)

    def test_no_heat_without_either_flow(self):
        for air, water in [(0, .001), (1, 0)]:
            self.assertEqual(reheat_coil(290, 340, air, water, 308.15).heat_to_air, 0)

    def test_cold_water_does_not_create_heat(self):
        self.assertLess(reheat_coil(300, 290, 1, .001, 308.15).heat_to_air, 0)

    def test_uniform_temperature_is_equilibrium(self):
        p = ZoneThermalParameters(1e5, 1e6, 10, 100, 20)
        self.assertEqual(zone_derivatives(p, 295, 295, 295, 295, 1), (0, 0))

    def test_internal_exchange_conserves_total_energy(self):
        p = ZoneThermalParameters(1e5, 1e6, 0, 100, 0)
        air, mass = zone_derivatives(p, 290, 300, 280, 290, 0)
        self.assertGreater(air, 0)
        self.assertLess(mass, 0)
        self.assertAlmostEqual(p.air_capacity * air + p.mass_capacity * mass, 0)

    def test_air_exchange_has_correct_sign_and_additive_energy(self):
        p = ZoneThermalParameters(1e5, 1e6, 0, 0, 0)
        cool, _ = zone_derivatives(p, 295, 295, 295, 285, 1)
        heat, _ = zone_derivatives(p, 295, 295, 295, 305, 1)
        self.assertAlmostEqual(cool, -heat)
        combined, _ = zone_derivatives(p, 295, 295, 295, 285, 1, reheat_power=5000)
        self.assertAlmostEqual(combined, cool + 5000 / p.air_capacity)

    def test_invalid_physical_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            ZoneThermalParameters(0, 1, 0, 0, 0)
        with self.assertRaises(ValueError):
            air_flows([1.2], [2], 200, 2000, 20, running=True)
        with self.assertRaises(ValueError):
            air_handler(295, 280, 1, 290, 1, 200, 0, running=True)


if __name__ == '__main__':
    unittest.main()
