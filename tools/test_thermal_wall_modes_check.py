"""DC-preserving, conservative wall-mode reduction and general capacity solve."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture, solve_reference
from tools.thermal_reduction_check import forcing
from tools.thermal_wall_modes_check import ModalResponse, check, transfer_audit, wall_basis


class ThermalWallModesCheckTest(unittest.TestCase):
    def test_basis_contains_constant_and_has_unit_projected_capacity(self):
        grid = fixture(room_rows=6, room_columns=5)
        for count in (0, 4, 43):
            basis, _ = wall_basis(grid, count)
            self.assertEqual(basis.shape[1], 121 + count)
            np.testing.assert_allclose(basis.T @ (grid.capacity[:, None] * basis),
                                      np.eye(basis.shape[1]), atol=1e-10)
            constant = basis @ (basis.T @ grid.capacity)
            np.testing.assert_allclose(constant, 1, atol=1e-10)

    def test_mode_spaces_are_nested_and_preserve_dc_ports(self):
        grid = fixture(room_rows=6, room_columns=5)
        bases = {str(n): wall_basis(grid, n)[0] for n in (0, 4, 43)}
        for coarse, fine in zip(bases.values(), list(bases.values())[1:]):
            np.testing.assert_allclose(fine @ (fine.T @ (grid.capacity[:, None] * coarse)), coarse, atol=1e-10)
        result = transfer_audit(grid, bases)
        for name, groups in result.items():
            for key, metrics in groups.items():
                if key.endswith('__dc') or name == '43':
                    self.assertLess(metrics['maximum_relative_column_error'], 1e-8)

    def test_full_enrichment_and_dense_galerkin_agree(self):
        grid = fixture(room_rows=6, room_columns=5)
        initial = np.linspace(285., 305., len(grid.capacity))
        for count in (0, 4, 43):
            basis, _ = wall_basis(grid, count)
            model = ModalResponse(grid, basis)
            state = model.initial(grid, initial)
            projected = basis @ state
            inputs = forcing('mixed', 0)
            actual = basis @ model.advance(state, inputs)
            expected = solve_reference(grid, projected, basis=basis, **inputs)
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
            if count == 43:
                np.testing.assert_allclose(actual, solve_reference(grid, initial, **inputs), atol=1e-8, rtol=0)

    def test_short_response_energy_and_full_enrichment(self):
        result = check(steps=4)
        for case in result['cases']:
            self.assertLess(case['reductions']['wall_modes_43']['maximum_zone_error_K'], 1e-8)
            for metrics in case['reductions'].values():
                self.assertLess(metrics['maximum_total_energy_residual_W'], 1e-7)

    def test_budget_rejected_before_fixture(self):
        with patch('tools.thermal_wall_modes_check.fixture') as mocked:
            for value in (0, 289, 1.5):
                with self.assertRaises(ValueError):
                    check(steps=value)
            mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
