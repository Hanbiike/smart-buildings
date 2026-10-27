"""Small source-contract fixtures, not environment rollouts or policy screens.

No scenario configurations, hidden building parameters or archive targets.
"""

from types import SimpleNamespace
import unittest

import numpy as np

from smart_buildings.smart_control.simulator import building, finite_difference_solver
from smart_buildings.smart_control.simulator.simulator_flexible_floor_plan import SimulatorFlexibleGeometries


class SimulatorSourceContractTest(unittest.TestCase):
    def test_air_stream_uses_diffusers_but_senses_room_mean(self):
        rooms = {'room_1': [(0, 0), (0, 1)], 'room_2': [(0, 2)]}
        field = SimpleNamespace(room_dict=rooms, temp=np.full((1, 3), 295.0),
            diffusers=np.array([[1.0, 0.0, 1.0]]),
            get_zone_average_temps=lambda: {'room_1': 295.0, 'room_2': 295.0})
        captured = []
        sim = SimulatorFlexibleGeometries.__new__(SimulatorFlexibleGeometries)
        sim.building = field
        sim._hvac = SimpleNamespace(vavs={'VAV_1': object()},
            get_zones_for_vav=lambda _: ['room_1', 'room_2'])
        sim._solver = SimpleNamespace(couple_air_side=lambda p, s: captured.append((p, s)))
        sim._couple_air_side()
        applied, sensed = [x.toarray().ravel() for x in captured[0]]
        np.testing.assert_allclose(applied, [2 / 3, 0, 1 / 3])
        np.testing.assert_allclose(sensed, [1 / 3] * 3)
        self.assertAlmostEqual(applied.sum(), 1)
        self.assertAlmostEqual(sensed.sum(), 1)

    def test_reheat_and_internal_power_are_uniform_not_diffuser_weighted(self):
        field = SimpleNamespace(input_q=np.zeros((1, 3)),
            diffusers=np.array([[1.0, 0, 0]]),
            _zone_indices=lambda _: (np.array([0, 0, 0]), np.array([0, 1, 2])))
        building.FloorPlanBasedBuilding.apply_thermal_power_zone(field, 'room_1', 900)
        np.testing.assert_allclose(field.input_q, [[300, 300, 300]])
        # It assigns rather than accumulates: caller must sum all sources first.
        building.FloorPlanBasedBuilding.apply_thermal_power_zone(field, 'room_1', 600)
        np.testing.assert_allclose(field.input_q, [[200, 200, 200]])

    def test_mass_capacity_and_coupling_follow_volume_and_height(self):
        material = building.MaterialProperties(conductivity=12, heat_capacity=1000, density=30)
        mass = building.InteriorMass(material, np.array([[True, False]]))
        capacity, conductance = finite_difference_solver.get_interior_mass_terms(
            mass, np.full((1, 2), .2), np.full((1, 2), .2), 3)
        np.testing.assert_allclose(capacity, [[3600, 0]])
        np.testing.assert_allclose(conductance, [[.16, 0]])
        self.assertAlmostEqual(capacity[0, 0] / conductance[0, 0], 22500)

    def test_zone_mean_alone_does_not_close_wall_heat_balance(self):
        # Two air cells and one wall; only the second air cell touches the wall.
        # Both starts have identical air mean and wall temperature, yet evolve
        # to different means. A zone-mean + one-mass state is not exact here.
        capacity_rate = np.diag([1., 1., 10.])
        laplacian = np.array([[1., -1., 0.], [-1., 3., -2.], [0., -2., 2.]])
        matrix = capacity_rate + laplacian
        first = np.array([290., 300., 280.])
        second = np.array([300., 290., 280.])
        self.assertEqual(first[:2].mean(), second[:2].mean())
        a = np.linalg.solve(matrix, capacity_rate @ first)
        b = np.linalg.solve(matrix, capacity_rate @ second)
        self.assertGreater(abs(a[:2].mean() - b[:2].mean()), .1)


if __name__ == '__main__':
    unittest.main()
