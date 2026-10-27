"""Pooled-prior fitting, causal adaptation and building-exclusion tests."""

import unittest
from unittest.mock import patch
import json

import numpy as np
from scipy.optimize import lsq_linear

from smart_buildings.baselines.pooled_greybox import (
    DEFAULT_PRIOR, FEATURE_LOWER, FEATURE_UPPER, PooledGreyBoxModel,
    fit_population_prior,
)
from tools.pooled_greybox_check import CaseExamples, collect_case, fit_fold, forecast_gate, summarize
from tools.test_causal_greybox import case


class PooledGreyboxTest(unittest.TestCase):
    def example(self, scenario, scale=1.0):
        rng = np.random.default_rng(123)
        x = rng.normal(size=(20, 8))
        theta = np.array([0.02, 0.1, 0.2, 0.15, 0.01, -0.01, 0.4, 0.03])
        return CaseExamples(scenario, x, scale * (x @ theta), 0)

    def test_sufficient_statistics_match_full_weighted_bounded_fit(self):
        a, b = self.example('a/weather'), self.example('b/weather', 0.7)
        fitted = fit_population_prior([(a.x, a.y), (b.x, b.y)])
        x = np.vstack([a.x / np.sqrt(len(a.x)), b.x / np.sqrt(len(b.x)), 0.1 * np.eye(8)])
        y = np.r_[a.y / np.sqrt(len(a.x)), b.y / np.sqrt(len(b.x)), 0.1 * DEFAULT_PRIOR]
        direct = lsq_linear(x, y, bounds=(FEATURE_LOWER, FEATURE_UPPER)).x
        np.testing.assert_allclose(fitted, direct, atol=1e-7)

    def test_case_weight_does_not_depend_on_room_or_row_count(self):
        a = self.example('a/weather')
        normal = fit_population_prior([(a.x, a.y)])
        repeated = fit_population_prior([(np.repeat(a.x, 3, axis=0), np.repeat(a.y, 3))])
        np.testing.assert_allclose(normal, repeated, atol=1e-10)

    def test_all_weather_cases_of_test_building_are_excluded(self):
        examples = [self.example('held/weather1'), self.example('held/weather2'),
                    self.example('other1/weather1'), self.example('other2/weather1', 0.7)]
        original, manifest = fit_fold(examples, 'held')
        for item in examples[:2]:
            item.x[:] = np.nan
            item.y[:] = np.inf
        changed, _ = fit_fold(examples, 'held')
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(manifest['training_buildings'], ['other1', 'other2'])
        self.assertTrue(all(not name.startswith('held/')
                            for name in manifest['training_scenarios']))

    def test_fold_requires_a_real_test_building_and_other_training_buildings(self):
        examples = [self.example('held/weather')]
        for group in ['held', 'absent']:
            with self.assertRaises(ValueError):
                fit_fold(examples, group)

    def test_invalid_training_arrays_and_ridge_are_rejected(self):
        for examples in [[], [(np.zeros((3, 7)), np.zeros(3))],
                         [(np.zeros((3, 8)), np.zeros(2))],
                         [(np.full((3, 8), np.nan), np.zeros(3))]]:
            with self.assertRaises(ValueError):
                fit_population_prior(examples)
        for ridge in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                fit_population_prior([], ridge_strength=ridge)

    def test_frozen_population_coefficients_are_copied_and_do_not_adapt(self):
        metadata, obs, action = case()
        prior = DEFAULT_PRIOR.copy()
        model = PooledGreyBoxModel(metadata, population_prior=prior, fit_window=12, adapt=False)
        prior[:] = 0
        for t in range(30):
            current = obs.copy()
            current[0] += t * 0.001
            model.update(current)
            model.set_action(*action)
        np.testing.assert_array_equal(model.coefficients[1], DEFAULT_PRIOR)
        self.assertTrue(np.isfinite(model.predict(1, 12)))
        self.assertFalse(model.trusted(1))

    def test_adaptation_changes_only_the_local_copy(self):
        metadata, obs, action = case()
        prior = DEFAULT_PRIOR.copy()
        model = PooledGreyBoxModel(metadata, population_prior=prior, fit_window=12, adapt=True)
        for t in range(30):
            current = obs.copy()
            current[0] += t * 0.001
            model.update(current)
            model.set_action(*action)
        self.assertFalse(np.allclose(model.coefficients[1], prior))
        np.testing.assert_array_equal(prior, DEFAULT_PRIOR)
        self.assertEqual(model.ridge_strength, 16)

    def test_unrelated_identity_metadata_does_not_change_predictions(self):
        metadata, obs, action = case()
        a = PooledGreyBoxModel(dict(metadata, building_id='first', scenario_id='cold'),
                               population_prior=DEFAULT_PRIOR, adapt=False)
        b = PooledGreyBoxModel(dict(metadata, building_id='other', scenario_id='hot'),
                               population_prior=DEFAULT_PRIOR, adapt=False)
        for model in (a, b):
            model.update(obs)
            model.set_action(*action)
        self.assertEqual(a.predict(1, 12), b.predict(1, 12))

    def test_population_fit_never_mutates_training_arrays(self):
        a = self.example('a/weather')
        x, y = a.x.copy(), a.y.copy()
        fit_population_prior([(a.x, a.y)])
        np.testing.assert_array_equal(a.x, x)
        np.testing.assert_array_equal(a.y, y)

    def test_coefficient_bounds_and_local_ridge_are_validated(self):
        metadata, _, _ = case()
        for prior in [np.zeros(7), np.full(8, np.nan), np.full(8, 100)]:
            with self.assertRaises(ValueError):
                PooledGreyBoxModel(metadata, population_prior=prior)
        with self.assertRaises(ValueError):
            PooledGreyBoxModel(metadata, population_prior=DEFAULT_PRIOR, local_ridge_strength=0)

    def test_collection_budget_is_checked_before_files_are_read(self):
        for offset, steps in [(0, 289), (-1, 288), (0, 1)]:
            with self.assertRaises(ValueError):
                collect_case('missing/weather', 'dsat_gen31', offset, steps)

    def test_collection_does_not_consume_a_preceding_window(self):
        metadata, obs, action = case()
        observations = np.tile(obs, (64, 1))
        observations[:32, :] = np.nan
        observations[32:, 0] += np.arange(32) * 0.0001
        dump = dict(obs=observations, action_continuous=np.tile(action[0], (64, 1)),
                    action_discrete=np.tile(action[1], (64, 1)))
        with patch('tools.pooled_greybox_check.Path.read_text', return_value=json.dumps(metadata)), \
                patch('tools.pooled_greybox_check.np.load') as load:
            load.return_value.__enter__.return_value = dump
            collected = collect_case('held/weather', 'dsat_gen31', 32, 32)
        self.assertEqual(collected.x.shape, (31, 8))
        self.assertTrue(np.isfinite(collected.x).all())
        self.assertTrue(np.isfinite(collected.y).all())
        self.assertIsNone(collected.occupied_observations)

    def test_summary_weights_cases_equally_not_by_zone_count(self):
        rows = []
        for value in [1.0, 3.0]:
            rows.append(dict(model='pooled', h6=value, h6_persist=4.0, h6_skill=1 - value / 4,
                             h6_regimes={'active': dict(model=value, persist=4.0, samples=100)},
                             h6_occupancy_regimes={}))
        report = summarize(rows, [6])['pooled']['h6']
        self.assertEqual(report['model_mae_K'], 2)
        self.assertEqual(report['relative_reduction'], 0.5)
        self.assertIsNone(report['occupied'])

    def test_forecast_gate_requires_active_and_occupied_improvement(self):
        rows = []
        for building in ['a', 'b']:
            entry = dict(model=0.5, persist=1.0, samples=100)
            rows.append(dict(model='pooled', scenario=f'{building}/weather',
                             h6=0.5, h6_persist=1.0,
                             h6_regimes={'active': entry.copy()},
                             h6_occupancy_regimes={'occupied': entry.copy()}))
        self.assertTrue(forecast_gate(rows, [6])['pooled']['passed'])
        rows[0]['h6_regimes']['active']['model'] = 1.1
        self.assertFalse(forecast_gate(rows, [6])['pooled']['passed'])

    def test_forecast_gate_rejects_missing_occupied_coverage(self):
        row = dict(model='pooled', scenario='a/weather', h6=0.5, h6_persist=1.0,
                   h6_regimes={'active': dict(model=0.5, persist=1.0, samples=100)},
                   h6_occupancy_regimes={})
        self.assertFalse(forecast_gate([row], [6])['pooled']['passed'])

    def test_forecast_gate_rejects_one_building_regression_despite_good_average(self):
        rows = []
        for building, error in [('a', 1.1), ('b', 0.1), ('c', 0.1)]:
            entry = dict(model=error, persist=1.0, samples=100)
            rows.append(dict(model='pooled', scenario=f'{building}/weather',
                             h6=error, h6_persist=1.0,
                             h6_regimes={'active': entry.copy()},
                             h6_occupancy_regimes={'occupied': entry.copy()}))
        self.assertFalse(forecast_gate(rows, [6])['pooled']['passed'])


if __name__ == '__main__':
    unittest.main()
