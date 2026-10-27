"""Input-response bases retain DC/energy and do not depend on response paths."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture
from tools.thermal_wall_modes_check import transfer_audit
from tools.thermal_port_modes_check import DESIGN_PERIODS, check, port_basis


class ThermalPortModesCheckTest(unittest.TestCase):
    def test_basis_is_nested_and_conservative(self):
        grid = fixture(room_rows=6, room_columns=5)
        bases = [port_basis(grid, count)[0] for count in (0, 4, 8, 16)]
        for basis in bases:
            np.testing.assert_allclose(basis.T @ (grid.capacity[:, None] * basis),
                                      np.eye(basis.shape[1]), atol=1e-10)
            np.testing.assert_allclose(basis @ (basis.T @ grid.capacity), 1, atol=1e-10)
        for coarse, fine in zip(bases, bases[1:]):
            np.testing.assert_allclose(fine @ (fine.T @ (grid.capacity[:, None] * coarse)), coarse, atol=1e-10)

    def test_dc_is_preserved_also_at_off_design_feedback(self):
        grid = fixture(room_rows=6, room_columns=5)
        basis, metadata = port_basis(grid, 8)
        result = transfer_audit(grid, {'port': basis}, periods=(None,), feedback=[('off', (1., 5.))])
        self.assertLess(result['port']['off__dc']['maximum_relative_column_error'], 1e-8)
        self.assertEqual(metadata['design_periods_seconds'], list(DESIGN_PERIODS))

    def test_basis_construction_never_reads_response_protocol(self):
        grid = fixture(room_rows=6, room_columns=5)
        with patch('tools.thermal_port_modes_check.forcing', side_effect=AssertionError('lookahead')):
            basis, metadata = port_basis(grid, 8)
        self.assertEqual(basis.shape[1], 129)
        self.assertGreater(metadata['numerical_snapshot_rank'], 0)

    def test_short_response_conserves_energy(self):
        result = check(steps=3)
        for case in result['cases']:
            for metrics in case['reductions'].values():
                self.assertLess(metrics['maximum_total_energy_residual_W'], 1e-7)

    def test_invalid_budget_is_rejected_before_fixture(self):
        with patch('tools.thermal_port_modes_check.fixture') as mocked:
            for steps in (0, 289, 1.5):
                with self.assertRaises(ValueError):
                    check(steps=steps)
            mocked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
