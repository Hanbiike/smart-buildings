"""Dense reference/source agreement and explicitly conservative reduction."""

from copy import copy
import unittest

import numpy as np

from tools.thermal_grid_reference import (
    energy_audit, fixture, reduced_basis, solve_reference, solve_source,
)


class ThermalGridReferenceTest(unittest.TestCase):
    def test_dense_assembly_matches_source_with_nonuniform_hidden_history(self):
        grid = fixture()
        previous = np.random.default_rng(42).uniform(280, 305, len(grid.capacity))
        for arguments in ({}, dict(conductance=(0., 0.)),
                          dict(ambient=300., supply=(303., 284.), fixed=(0., 7.))):
            with self.subTest(arguments=arguments):
                np.testing.assert_allclose(solve_source(grid, previous, **arguments),
                    solve_reference(grid, previous, **arguments), rtol=0, atol=1e-9)

    def test_uniform_equilibrium_is_a_fixed_point(self):
        grid = fixture()
        previous = np.full(len(grid.capacity), 295.)
        arguments = dict(ambient=295., supply=(295., 295.), fixed=(0., 0.))
        for basis in (None, reduced_basis(grid), np.eye(len(previous))):
            np.testing.assert_allclose(solve_reference(grid, previous, basis=basis, **arguments),
                previous, rtol=0, atol=1e-9)

    def test_full_energy_and_each_group_close_and_flux_is_antisymmetric(self):
        grid = fixture()
        previous = np.linspace(285, 302, len(grid.capacity))
        audit = energy_audit(grid, previous, solve_reference(grid, previous))
        self.assertLess(abs(audit['total_balance_residual_W']), 1e-7)
        self.assertLess(audit['maximum_node_balance_residual_W'], 1e-7)
        np.testing.assert_allclose(audit['group_balance_residual_W'], 0, atol=1e-7)
        flux = np.array(audit['group_exchange_W'])
        np.testing.assert_allclose(flux + flux.T, 0, atol=1e-10)
        # Explicit shared-wall path, not an invented direct room-room link.
        self.assertGreater(abs(flux[0, 4]), 0)
        self.assertGreater(abs(flux[2, 4]), 0)
        self.assertEqual(flux[0, 2], 0)

    def test_reduction_conserves_group_energy_but_not_cell_equations(self):
        grid = fixture()
        previous = np.full(len(grid.capacity), 295.)
        basis = reduced_basis(grid)
        np.testing.assert_array_equal(basis.sum(axis=1), 1)
        np.testing.assert_array_equal(basis.sum(axis=0) > 0, True)
        full = solve_reference(grid, previous)
        reduced = solve_reference(grid, previous, basis=basis)
        audit = energy_audit(grid, previous, reduced)
        self.assertLess(abs(audit['total_balance_residual_W']), 1e-7)
        np.testing.assert_allclose(audit['group_balance_residual_W'], 0, atol=1e-7)
        self.assertGreater(audit['maximum_node_balance_residual_W'], 1)
        self.assertGreater(np.max(np.abs(grid.sensed.T @ (reduced - full))), 1e-3)

    def test_air_application_map_changes_transient_even_at_equal_total_power_weights(self):
        grid = fixture()
        previous = np.full(len(grid.capacity), 295.)
        altered = copy(grid)
        altered.applied = grid.uniform.copy()
        np.testing.assert_allclose(grid.applied.sum(axis=0), altered.applied.sum(axis=0))
        original = grid.sensed.T @ solve_reference(grid, previous)
        uniform = grid.sensed.T @ solve_reference(altered, previous)
        self.assertGreater(np.max(np.abs(original - uniform)), 1e-4)


if __name__ == '__main__':
    unittest.main()
