"""Small whole-system bases: DC, energy, independent design and recurrence."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture, solve_reference
from tools.thermal_wall_modes_check import ModalResponse, transfer_audit
from tools.thermal_system_modes_check import check, initial_state, stability_audit, system_basis


class ThermalSystemModesCheckTest(unittest.TestCase):
    def test_small_nested_bases_preserve_constant_and_capacity(self):
        grid = fixture(room_rows=6, room_columns=5)
        for adjoint in (False, True):
            bases = [system_basis(grid, n, adjoint=adjoint)[0] for n in (0, 8, 16, 24)]
            self.assertEqual([basis.shape[1] for basis in bases], [5, 13, 21, 29])
            for basis in bases:
                np.testing.assert_allclose(basis.T @ (grid.capacity[:, None] * basis),
                                          np.eye(basis.shape[1]), atol=1e-10)
                np.testing.assert_allclose(basis @ (basis.T @ grid.capacity), 1, atol=1e-10)
            for coarse, fine in zip(bases, bases[1:]):
                np.testing.assert_allclose(fine @ (fine.T @ (grid.capacity[:, None] * coarse)), coarse, atol=1e-10)

    def test_off_design_dc_and_sampled_stability(self):
        grid = fixture(room_rows=6, room_columns=5)
        basis, _ = system_basis(grid, 8, adjoint=True)
        result = transfer_audit(grid, {'small': basis}, periods=(None,), feedback=[('off', (1., 5.))])
        self.assertLess(result['small']['off__dc']['maximum_relative_column_error'], 1e-8)
        for radius in stability_audit(ModalResponse(grid, basis)).values():
            self.assertLess(radius, 1)

    def test_design_never_reads_initial_or_time_response_functions(self):
        grid = fixture(room_rows=6, room_columns=5)
        with patch('tools.thermal_system_modes_check.initial_state', side_effect=AssertionError('initial leak')):
            with patch('tools.thermal_system_modes_check.input_at', side_effect=AssertionError('forcing leak')):
                basis, _ = system_basis(grid, 8, adjoint=True)
        self.assertEqual(basis.shape[1], 13)

    def test_general_projection_recurrence_and_unseen_initial_condition(self):
        grid = fixture(room_rows=6, room_columns=5)
        initial = initial_state(grid, 'initial_only')
        self.assertGreater(np.std(initial), 0)
        basis, _ = system_basis(grid, 8, adjoint=True)
        model = ModalResponse(grid, basis)
        state = model.initial(grid, initial)
        field = basis @ state
        for _ in range(3):
            expected = solve_reference(grid, field, basis=basis)
            state = model.advance(state, dict(ambient=275., fixed=(3., 2.), conductance=(4., 3.), supply=(285., 289.)))
            field = basis @ state
            np.testing.assert_allclose(field, expected, atol=1e-9, rtol=0)

    def test_short_check_energy_and_initial_only_case(self):
        result = check(steps=3)
        self.assertIn('initial_only', [row['protocol'] for row in result['cases']])
        for case in result['cases']:
            for metrics in case['reductions'].values():
                self.assertLess(metrics['maximum_total_energy_residual_W'], 1e-7)

    def test_invalid_budget_fails_early(self):
        with patch('tools.thermal_system_modes_check.fixture') as mocked:
            for steps in (0, 289, 2.5):
                with self.assertRaises(ValueError):
                    check(steps=steps)
            mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
