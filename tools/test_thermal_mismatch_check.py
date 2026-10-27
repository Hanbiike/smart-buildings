"""Frozen nominal observer is separated from perturbed plant and clean targets."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture
from tools.thermal_system_modes_check import system_basis
from tools.thermal_wall_modes_check import ModalResponse
from tools.thermal_state_observer_check import check as nominal_check, generate_data
from tools.thermal_mismatch_check import (
    Mismatch, check, evaluate_received, generate_mismatch, perturb_network, realized_input,
)


class ThermalMismatchCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = fixture(room_rows=6, room_columns=5)
        cls.model = ModalResponse(cls.grid, system_basis(cls.grid, 8, adjoint=True)[0])

    def test_perturbation_is_passive_and_does_not_mutate_nominal(self):
        before = self.grid.laplacian.copy()
        truth = perturb_network(self.grid, Mismatch('test', capacity_scale=1.2, wall_link_scale=.8))
        np.testing.assert_array_equal(self.grid.laplacian, before)
        np.testing.assert_allclose(truth.laplacian.sum(axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(truth.laplacian, truth.laplacian.T)
        self.assertTrue(np.all(truth.capacity > 0))
        np.testing.assert_allclose(truth.capacity, self.grid.capacity * 1.2)
        self.assertGreater(np.linalg.norm(truth.laplacian - before), 0)

    def test_actual_airflow_and_hidden_heat_never_modify_command(self):
        command = dict(ambient=295., fixed=(3., 2.), conductance=(4., 3.), supply=(280., 295.))
        spec = Mismatch('hidden', airflow_scale=.8, hidden_room_1_heat_W=5.)
        actual = realized_input(command, 96, spec)
        self.assertEqual(command['fixed'], (3., 2.))
        self.assertEqual(actual['fixed'], (8., 2.))
        np.testing.assert_allclose(actual['conductance'], [3.2, 2.4])
        self.assertEqual(realized_input(command, 95, spec)['fixed'], command['fixed'])

    def test_noise_does_not_change_clean_targets_and_reset_duplicate(self):
        clean_nominal, _ = generate_data(self.grid, 'mixed', 24)
        received, clean, _ = generate_mismatch(self.grid, 'mixed', 24, Mismatch('noise', sensor_noise_std_K=.05))
        np.testing.assert_allclose(clean, clean_nominal, atol=1e-9)
        np.testing.assert_array_equal(received[0], received[1])
        self.assertGreater(np.std(received - clean), .01)

    def test_truth_range_is_optional_generation_diagnostic_only(self):
        diagnostics = {}
        received, clean, commands = generate_mismatch(self.grid, 'mixed', 24,
            Mismatch('nominal'), diagnostics=diagnostics)
        self.assertIn('true_physical_temperature_range_K', diagnostics)
        low, high = diagnostics['true_physical_temperature_range_K']
        self.assertTrue(np.all(clean >= low - 1e-10))
        self.assertTrue(np.all(clean <= high + 1e-10))
        # evaluate_received has no diagnostics/true-parameter argument.
        row = evaluate_received(self.grid, self.model, received, clean, commands, prefix=6)
        self.assertNotIn('true_physical_temperature_range_K', row)

    def test_clean_targets_are_evaluation_only_and_coefficients_frozen(self):
        received, clean, commands = generate_mismatch(self.grid, 'mixed', 24, Mismatch('noise', sensor_noise_std_K=.05))
        before = self.model.k.copy()
        first = evaluate_received(self.grid, self.model, received, clean, commands, prefix=6)
        changed = clean.copy()
        changed[8:] += 20
        second = evaluate_received(self.grid, self.model, received, changed, commands, prefix=6)
        self.assertEqual(first['reconstructed_temperature_range_K'], second['reconstructed_temperature_range_K'])
        self.assertEqual(first['maximum_post_update_sensor_residual_K'], second['maximum_post_update_sensor_residual_K'])
        self.assertNotEqual(first['metrics']['all'], second['metrics']['all'])
        np.testing.assert_array_equal(self.model.k, before)

    def test_nominal_arm_reproduces_prior_observer_protocol(self):
        received, clean, commands = generate_mismatch(self.grid, 'mixed', 24, Mismatch('nominal'))
        result = evaluate_received(self.grid, self.model, received, clean, commands, prefix=6)
        reference = nominal_check(steps=24, prefix=6)
        row = next(row for row in reference['cases'] if row['protocol'] == 'mixed' and row['states'] == 13)
        self.assertEqual(result['metrics']['all'], row['metrics'])

    def test_invalid_budget_precedes_fixture_and_bad_scales_fail(self):
        with patch('tools.thermal_mismatch_check.fixture') as mocked:
            for options in (dict(steps=289), dict(steps=20, prefix=19), dict(prefix=-1)):
                with self.assertRaises(ValueError):
                    check(**options)
            mocked.assert_not_called()
        for spec in (Mismatch('bad', capacity_scale=0), Mismatch('bad', sensor_noise_std_K=np.nan)):
            with self.assertRaises(ValueError):
                perturb_network(self.grid, spec)


if __name__ == '__main__':
    unittest.main()
