"""Checker guards must reject invalid budgets before accessing any dump."""

import unittest
from unittest.mock import patch
import json

import numpy as np

from tools.greybox_check import evaluate, make_model
from tools.test_causal_greybox import case


class GreyboxCheckerTest(unittest.TestCase):
    def test_budget_is_enforced_for_direct_python_calls(self):
        for steps in (0, 289, 8640):
            with self.assertRaises(ValueError):
                evaluate('does-not-exist', steps=steps)

    def test_warmup_and_horizons_are_validated_before_loading(self):
        for kwargs in (dict(fit_window=288), dict(horizons=()),
                       dict(horizons=(0,)), dict(horizons=(200,)), dict(offset=-1)):
            with self.assertRaises(ValueError):
                evaluate('does-not-exist', **kwargs)

    def test_oracle_diagnostic_is_not_supported_by_observation_only_models(self):
        for kind in ('legacy', 'sensors'):
            with self.assertRaises(ValueError):
                evaluate('does-not-exist', model_kind=kind, oracle_actions=True)

    def test_model_factory_keeps_ablations_separate(self):
        metadata, _, _ = case()
        for kind, water, mass in [('water', True, False), ('mass', False, True),
                                  ('thermal', True, True)]:
            model = make_model(metadata, kind, 'batch_rls', 144)
            self.assertEqual(model.physical_water, water)
            self.assertEqual(model.thermal_mass, mass)
            self.assertFalse(model.trusted(1))
        with self.assertRaises(ValueError):
            make_model(metadata, 'unknown', 'batch_rls', 144)

    def test_report_is_strict_json_serializable(self):
        metadata, obs, action = case()
        for drift in (0.0, 0.0001):
            observations = np.tile(obs, (32, 1))
            observations[:, 0] += np.arange(32) * drift
            dump = dict(obs=observations, action_continuous=np.tile(action[0], (32, 1)),
                        action_discrete=np.tile(action[1], (32, 1)))
            with patch('tools.greybox_check.Path.read_text', return_value=json.dumps(metadata)), \
                    patch('tools.greybox_check.np.load') as load:
                load.return_value.__enter__.return_value = dump
                report = evaluate('synthetic/case', steps=32, fit_window=12,
                                  horizons=(1, 3), model_kind='mass')
            json.dumps(report, allow_nan=False)
            self.assertIsInstance(report['prefix_excitation']['air_excited_rooms'], int)
            if drift == 0:
                self.assertIsNone(report['h1_skill'])

    def test_offset_starts_fresh_and_requires_a_full_window(self):
        metadata, obs, action = case()
        observations = np.tile(obs, (64, 1))
        observations[:32, 0] = -10  # Must never be consumed as a warm-up.
        observations[32:, 0] += np.arange(32) * 0.0001
        dump = dict(obs=observations, action_continuous=np.tile(action[0], (64, 1)),
                    action_discrete=np.tile(action[1], (64, 1)))
        with patch('tools.greybox_check.Path.read_text', return_value=json.dumps(metadata)), \
                patch('tools.greybox_check.np.load') as load:
            load.return_value.__enter__.return_value = dump
            shifted = evaluate('synthetic/case', steps=32, fit_window=12,
                               horizons=(1, 3), model_kind='mass', offset=32)
            for key in dump:
                dump[key] = dump[key][32:]
            fresh = evaluate('synthetic/case', steps=32, fit_window=12,
                             horizons=(1, 3), model_kind='mass')
            self.assertEqual(shifted['h3'], fresh['h3'])
            with self.assertRaises(ValueError):
                evaluate('synthetic/case', steps=32, fit_window=12,
                         horizons=(1, 3), model_kind='mass', offset=1)


if __name__ == '__main__':
    unittest.main()
