"""Generalized reference and factorial spatial-reduction contracts."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture, solve_reference, solve_source
from tools.thermal_reduction_check import forcing
from tools.thermal_spatial_check import AIR_LEVELS, WALL_LEVELS, CompiledResponse, check, partitions


class ThermalSpatialCheckTest(unittest.TestCase):
    def test_larger_fixture_has_core_and_matches_source(self):
        grid = fixture(room_rows=6, room_columns=5)
        self.assertEqual(len(grid.capacity), 164)
        _, counts = partitions(grid, air_level='spatial', wall_level='coarse')
        self.assertEqual(counts, [12, 12])
        previous = np.random.default_rng(17).uniform(285, 300, len(grid.capacity))
        np.testing.assert_allclose(solve_source(grid, previous),
                                   solve_reference(grid, previous), atol=1e-9, rtol=0)

    def test_factorial_partitions_cover_and_refine_each_axis(self):
        grid = fixture(room_rows=6, room_columns=5)
        bases = {(air, wall): partitions(grid, air_level=air, wall_level=wall)[0]
                 for air in AIR_LEVELS for wall in WALL_LEVELS}
        self.assertEqual([bases[air, wall].shape[1] for air in AIR_LEVELS for wall in WALL_LEVELS],
                         [13, 20, 48, 29, 36, 64, 129, 136, 164])
        for basis in bases.values():
            np.testing.assert_array_equal(basis.sum(axis=1), 1)
            self.assertTrue(np.all(basis.sum(axis=0) > 0))
        pairs = [(bases[a, wall], bases[b, wall]) for wall in WALL_LEVELS
                 for a, b in zip(AIR_LEVELS, AIR_LEVELS[1:])]
        pairs += [(bases[air, a], bases[air, b]) for air in AIR_LEVELS
                  for a, b in zip(WALL_LEVELS, WALL_LEVELS[1:])]
        for coarse, fine in pairs:
            self.assertTrue(np.all(np.count_nonzero(fine.T @ coarse, axis=1) == 1))

    def test_compiled_response_matches_dense_projection_and_updates_own_state(self):
        grid = fixture(room_rows=6, room_columns=5)
        initial = np.linspace(290, 300, len(grid.capacity))
        basis, _ = partitions(grid, air_level='spatial', wall_level='segments')
        model = CompiledResponse(grid, basis)
        state = model.initial(grid, initial)
        field = basis @ state
        np.testing.assert_allclose(basis.T @ (grid.capacity * (field - initial)), 0, atol=1e-7)
        for step in (0, 24, 48):
            inputs = forcing('mixed', step)
            expected = solve_reference(grid, field, basis=basis, **inputs)
            state = model.advance(state, inputs)
            field = basis @ state
            np.testing.assert_allclose(field, expected, atol=1e-9, rtol=0)
        self.assertEqual(len(model.factors), 2)

    def test_short_response_conservation_and_explicit_protocol(self):
        report = check(steps=4)
        self.assertEqual(report['room_cells'], [30, 30])
        self.assertLess(report['maximum_first_step_source_reference_error_K'], 1e-9)
        for case in report['cases']:
            self.assertLess(case['reductions']['full__full']['maximum_zone_error_K'], 1e-9)
            for metrics in case['reductions'].values():
                self.assertLess(metrics['maximum_total_balance_residual_W'], 1e-7)
                self.assertLess(metrics['maximum_group_balance_residual_W'], 1e-7)
                self.assertIsNone(metrics['after_12_steps_zone_mae_K'])

    def test_invalid_budgets_and_geometries_are_rejected(self):
        with patch('tools.thermal_spatial_check.fixture') as mocked:
            for steps in (0, 289, 3.5):
                with self.assertRaises(ValueError):
                    check(steps=steps)
            mocked.assert_not_called()
        for options in (dict(room_rows=0), dict(room_columns=13), dict(room_rows=2.5)):
            with self.assertRaises(ValueError):
                fixture(**options)


if __name__ == '__main__':
    unittest.main()
