"""Nested conservative bases and bounded free-response protocol."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture
from tools.thermal_reduction_check import LEVELS, check, nested_basis, project_initial


class ThermalReductionCheckTest(unittest.TestCase):
    def test_partitions_are_complete_and_nested(self):
        grid = fixture()
        bases = [nested_basis(grid, level) for level in LEVELS]
        self.assertEqual([b.shape[1] for b in bases], [27, 31, 35, 47])
        for basis in bases:
            np.testing.assert_array_equal(basis.sum(axis=1), 1)
            self.assertTrue(np.all(basis.sum(axis=0) > 0))
        for coarse, fine in zip(bases, bases[1:]):
            # Every fine group lies entirely within one coarse group.
            incidence = fine.T @ coarse
            self.assertTrue(np.all(np.count_nonzero(incidence, axis=1) == 1))

    def test_initial_projection_preserves_group_energy(self):
        grid = fixture()
        initial = np.linspace(280., 300., len(grid.capacity))
        for level in LEVELS:
            basis = nested_basis(grid, level)
            projected = project_initial(grid, initial, basis)
            np.testing.assert_allclose(basis.T @ (grid.capacity * (projected - initial)), 0, atol=1e-8)

    def test_full_basis_recovers_free_response_and_conservation(self):
        result = check(steps=16)
        for case in result['cases']:
            full = case['reductions']['full']
            self.assertLess(full['maximum_zone_error_K'], 1e-8)
            for metrics in case['reductions'].values():
                self.assertLess(metrics['maximum_total_balance_residual_W'], 1e-7)
                self.assertLess(metrics['maximum_group_balance_residual_W'], 1e-7)

    def test_refined_partition_keeps_diffuser_separate(self):
        grid = fixture()
        for level in ('diffuser', 'boundary_side', 'full'):
            basis = nested_basis(grid, level)
            for room in range(2):
                cell = np.flatnonzero(grid.applied[:, room])[0]
                group = np.flatnonzero(basis[cell])[0]
                self.assertEqual(basis[:, group].sum(), 1)

    def test_invalid_budget_fails_before_fixture_creation(self):
        with patch('tools.thermal_reduction_check.fixture') as mocked:
            for steps in (0, 289, -1, 2.5):
                with self.assertRaises(ValueError):
                    check(steps=steps)
            mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
