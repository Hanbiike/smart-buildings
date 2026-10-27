"""Start binding happens before any environment construction or stepping."""

from pathlib import Path
import unittest
from unittest.mock import patch

from smart_buildings.evaluate import load_hybrid_environment


class EvaluationStartTest(unittest.TestCase):
    def test_timezone_conversion_and_binding_before_construction(self):
        events = []
        with patch('smart_buildings.evaluate.gin.clear_config'), \
             patch('smart_buildings.evaluate.gin.parse_config_file'), \
             patch('smart_buildings.evaluate.gin.bind_parameter', side_effect=lambda k, v: events.append((k, v))), \
             patch('smart_buildings.evaluate.hybrid_action_environment.HybridActionEnvironment', side_effect=lambda: events.append('constructed')):
            load_hybrid_environment(Path('unused'), start_timestamp='2023-01-03T06:00:00-08:00')
        self.assertEqual(events, [('sim/to_timestamp.date_str', '2023-01-03 14:00:00'), 'constructed'])


if __name__ == '__main__':
    unittest.main()
