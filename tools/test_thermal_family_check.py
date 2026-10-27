"""Finite-family selection uses prefixes only; envelopes make no confidence claim."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture
from tools.thermal_mismatch_check import Mismatch, generate_mismatch
from tools.thermal_prefix_identification_check import candidate
from tools.thermal_state_observer_check import DelayedObserver
from tools.thermal_system_modes_check import system_basis
from tools.thermal_family_check import (
    check, envelope_metrics, evaluate, family_vectors, forecast_distribution, select_family,
)


class ThermalFamilyCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = fixture(room_rows=6, room_columns=5)
        cls.basis = system_basis(cls.grid, 8, adjoint=True)[0]
        cls.received, cls.clean, cls.commands = generate_mismatch(cls.grid, 'mixed', 84, Mismatch('nominal'))
        cls.selection = select_family(cls.grid, cls.basis, cls.received[:48], cls.commands[:48])

    def test_declared_grid_and_positive_normalized_weights(self):
        vectors = family_vectors()
        self.assertEqual(len(vectors), 135)
        self.assertEqual(len({tuple(v) for v in vectors}), 135)
        self.assertTrue(any(np.array_equal(v, np.zeros(5)) for v in vectors))
        selected, weights, diagnostics = self.selection
        self.assertEqual(len(selected), diagnostics['retained'])
        self.assertTrue(np.all(weights > 0))
        self.assertAlmostEqual(weights.sum(), 1)

    def test_future_mutation_does_not_change_membership_or_weights(self):
        changed = self.received.copy()
        changed[48:] += 100
        commands = [dict(command) for command in self.commands]
        commands[48]['fixed'] = (10000., 0.)
        vectors, weights, diagnostics = select_family(self.grid, self.basis, changed[:48], commands[:48])
        np.testing.assert_array_equal(vectors, self.selection[0])
        np.testing.assert_array_equal(weights, self.selection[1])
        self.assertEqual(diagnostics, self.selection[2])

    def test_forecast_covariance_does_not_mutate_observer(self):
        grid, model, path = candidate(self.grid, self.basis, np.zeros(5), self.commands)
        observer = DelayedObserver(grid, model)
        observer.receive(self.received[0])
        observer.receive(self.received[1])
        state, covariance = observer.state.copy(), observer.p.copy()
        prediction, std = forecast_distribution(observer, path[:12])
        self.assertEqual(prediction.shape, (12, 2))
        self.assertTrue(np.all(std >= 0))
        np.testing.assert_array_equal(observer.state, state)
        np.testing.assert_array_equal(observer.p, covariance)

    def test_coverage_reports_misses_instead_of_claiming_guarantees(self):
        result = envelope_metrics([0., 1., 2.], [-.5, .5, 0.], [.5, 1.5, 1.])
        self.assertAlmostEqual(result['coverage_fraction'], 2 / 3)
        self.assertEqual(result['maximum_miss_K'], 1)
        self.assertEqual(result['mean_width_K'], 1)

    def test_clean_targets_affect_scores_not_states_and_augmentation_widens(self):
        vectors, weights, _ = self.selection
        vectors, weights = vectors[:2], weights[:2] / weights[:2].sum()
        saved_vectors, saved_weights = np.array(vectors).copy(), weights.copy()
        kwargs = dict(prefix=48, vectors=vectors, weights=weights, point=np.zeros(5))
        first = evaluate(self.grid, self.basis, self.received, self.clean, self.commands, **kwargs)
        shifted = self.clean.copy()
        shifted[48:] += 10
        second = evaluate(self.grid, self.basis, self.received, shifted, self.commands, **kwargs)
        np.testing.assert_array_equal(vectors, saved_vectors)
        np.testing.assert_array_equal(weights, saved_weights)
        self.assertEqual(first['weighted_reconstructed_field_range_K'], second['weighted_reconstructed_field_range_K'])
        self.assertNotEqual(first['metrics'], second['metrics'])
        for values in first['metrics']['all'].values():
            self.assertGreaterEqual(values['covariance_augmented_envelope']['mean_width_K'],
                                    values['family_envelope']['mean_width_K'])
            self.assertGreaterEqual(values['covariance_augmented_envelope']['coverage_fraction'],
                                    values['family_envelope']['coverage_fraction'])

    def test_budget_and_selection_prefix_reject_early(self):
        with patch('tools.thermal_family_check.fixture') as mocked:
            for options in (dict(steps=289), dict(prefix=73), dict(steps=40, prefix=30)):
                with self.assertRaises(ValueError):
                    check(**options)
            mocked.assert_not_called()
        with self.assertRaises(ValueError):
            select_family(self.grid, self.basis, self.received[:73], self.commands[:73])


if __name__ == '__main__':
    unittest.main()
