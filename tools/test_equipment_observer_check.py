"""Bounded-window and report tests for equipment-input validation."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from tools.equipment_observer_check import check
from tools.test_equipment_observer import observer_case


class EquipmentObserverCheckTest(unittest.TestCase):
    def test_invalid_budgets_are_rejected_before_file_access(self):
        for kwargs in [dict(steps=289), dict(steps=0), dict(offset=-1), dict(warmup=288)]:
            with self.assertRaises(ValueError):
                check('missing/case', **kwargs)

    def test_shifted_window_does_not_consume_nan_prefix(self):
        metadata, obs, action = observer_case()
        observations = np.tile(obs, (64, 1))
        observations[:32] = np.nan
        dump = dict(obs=observations, action_continuous=np.tile(action[0], (64, 1)),
                    action_discrete=np.tile(action[1], (64, 1)))
        with patch('tools.equipment_observer_check.Path.read_text', return_value=json.dumps(metadata)), \
                patch('tools.equipment_observer_check.Path.exists', return_value=False), \
                patch('tools.equipment_observer_check.np.load') as load:
            load.return_value.__enter__.return_value = dump
            report = check('synthetic/case', steps=32, offset=32, warmup=6)
            self.assertEqual(report['delayed_transitions'], 30)
            self.assertEqual(report['handler_flow_m3_s']['mae'], 0)
            self.assertEqual(report['water_sensor_K']['mae'], 0)
            self.assertIn('NOT validated', report['power_validation'])
            json.dumps(report, allow_nan=False)
            with self.assertRaises(ValueError):
                check('synthetic/case', steps=32, offset=33, warmup=6)

    def test_missing_capacity_is_reported_instead_of_filling_predictions(self):
        metadata, obs, action = observer_case()
        observations = np.tile(obs, (32, 1))
        observations[:, 13:15] = 0
        dump = dict(obs=observations, action_continuous=np.tile(action[0], (32, 1)),
                    action_discrete=np.tile(action[1], (32, 1)))
        with patch('tools.equipment_observer_check.Path.read_text', return_value=json.dumps(metadata)), \
                patch('tools.equipment_observer_check.Path.exists', return_value=False), \
                patch('tools.equipment_observer_check.np.load') as load:
            load.return_value.__enter__.return_value = dump
            report = check('synthetic/case', steps=32, offset=0, warmup=6)
        self.assertEqual(report['skipped_unobserved_capacity'], 25)
        self.assertEqual(report['handler_flow_m3_s']['samples'], 0)
        self.assertIsNone(report['handler_flow_m3_s']['mae'])
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
