"""Prefix-only fitting, excitation gates and no future-load identification."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture
from tools.thermal_mismatch_check import Mismatch, generate_mismatch
from tools.thermal_system_modes_check import system_basis
from tools.thermal_prefix_identification_check import NAMES, candidate, check, fit_prefix, prefix_residual


class ThermalPrefixIdentificationCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = fixture(room_rows=6, room_columns=5)
        cls.basis = system_basis(cls.grid, 8, adjoint=True)[0]

    def test_late_only_airflow_is_unidentified_not_filled_from_future(self):
        received, _, commands = generate_mismatch(self.grid, 'late_air_pulse', 112,
                                                  Mismatch('unknown', airflow_scale=.8))
        vector, diagnostics = fit_prefix(self.grid, self.basis, received[:72], commands[:72])
        np.testing.assert_array_equal(vector, 0)
        self.assertEqual(diagnostics['status'], 'no_prefix_excitation_or_sensitivity')
        self.assertIn('log_airflow', diagnostics['unidentified_parameters'])
        self.assertEqual(set(diagnostics['unidentified_parameters']), set(NAMES))

    def test_late_hidden_load_cannot_change_prefix_fit(self):
        results = []
        for spec in (Mismatch('nominal'), Mismatch('hidden', hidden_room_1_heat_W=5.)):
            received, _, commands = generate_mismatch(self.grid, 'mixed', 112, spec)
            results.append(fit_prefix(self.grid, self.basis, received[:48], commands[:48], max_evaluations=1))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        self.assertEqual(results[0][1], results[1][1])

    def test_future_observations_commands_do_not_affect_fitter(self):
        received, _, commands = generate_mismatch(self.grid, 'mixed', 84, Mismatch('nominal'))
        first = fit_prefix(self.grid, self.basis, received[:48], commands[:48], max_evaluations=1)
        changed = received.copy()
        changed[48:] += 100
        commands[48]['fixed'] = (10000., 0.)
        second = fit_prefix(self.grid, self.basis, changed[:48], commands[:48], max_evaluations=1)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_candidate_does_not_modify_nominal_and_uses_explicit_units(self):
        before = self.grid.capacity.copy()
        command = dict(ambient=295., fixed=(0., 1.), conductance=(4., 3.), supply=(280., 295.))
        grid, _, path = candidate(self.grid, self.basis, np.array([np.log(1.2), 0, np.log(.8), 1, 0]), [command])
        np.testing.assert_array_equal(self.grid.capacity, before)
        np.testing.assert_allclose(grid.capacity, before * 1.2)
        np.testing.assert_allclose(path[0]['conductance'], [3.2, 2.4])
        self.assertEqual(path[0]['fixed'], (5., 1.))
        self.assertEqual(command['fixed'], (0., 1.))

    def test_prefix_residual_shape_has_only_complete_in_prefix_targets(self):
        received, _, commands = generate_mismatch(self.grid, 'mixed', 72, Mismatch('nominal'))
        residual = prefix_residual(self.grid, self.basis, received[:72], commands[:72], np.zeros(5))
        # 8 origins, 3 horizons, 2 sensors; all forecast targets <= observation 66.
        self.assertEqual(residual.shape, (48,))
        self.assertTrue(np.isfinite(residual).all())

    def test_state_profile_cannot_increase_parameter_sensitivity(self):
        received, _, commands = generate_mismatch(self.grid, 'mixed', 72, Mismatch('nominal'))
        _, diagnostics = fit_prefix(self.grid, self.basis, received[:72], commands[:72], max_evaluations=1)
        self.assertEqual(diagnostics['initial_nuisance_directions'], 11)
        for name in NAMES:
            self.assertLessEqual(diagnostics['state_profiled_sensitivity_rms'][name],
                diagnostics['sensitivity_rms_K_per_coordinate'][name] + 1e-10)

    def test_budget_and_prefix_constraints_reject_early(self):
        with patch('tools.thermal_prefix_identification_check.fixture') as mocked:
            for options in (dict(steps=289), dict(prefix=73), dict(prefix=30.5), dict(steps=40, prefix=30)):
                with self.assertRaises(ValueError):
                    check(**options)
            mocked.assert_not_called()
        with self.assertRaises(ValueError):
            fit_prefix(self.grid, self.basis, np.zeros((73, 2)), [{}] * 73)


if __name__ == '__main__':
    unittest.main()
