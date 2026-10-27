"""Monitor calibration is frozen and rejection is causal and latched."""

import unittest
import numpy as np
from tools.thermal_adequacy_check import AdequacyMonitor
from tools.thermal_adequacy_check import run_monitor
from tools.thermal_grid_reference import fixture
from tools.thermal_system_modes_check import system_basis
from tools.thermal_mismatch_check import Mismatch, generate_mismatch


class AdequacyMonitorTest(unittest.TestCase):
    def monitor(self):
        monitor = AdequacyMonitor()
        for _ in range(24):
            monitor.calibrate([295, 295], [294.999, 294.999], [295.001, 295.001])
        monitor.freeze()
        return monitor

    def test_nominal_and_latched_rejection(self):
        monitor = self.monitor()
        self.assertEqual(monitor.observe(72, [295, 295], [294.999]*2, [295.001]*2)['status'], 'HEALTHY')
        self.assertEqual(monitor.observe(98, [296, 295], [294.999]*2, [295.001]*2)['first_alarm'], 98)
        self.assertEqual(monitor.observe(99, [295, 295], [294.999]*2, [295.001]*2)['status'], 'UNTRUSTED')

    def test_threshold_cannot_change_after_calibration(self):
        monitor = self.monitor()
        before = monitor.limit.copy()
        monitor.observe(98, [300, 295], [294.999]*2, [295.001]*2)
        np.testing.assert_array_equal(monitor.limit, before)
        with self.assertRaises(ValueError):
            monitor.calibrate([300, 295], [294]*2, [296]*2)
        with self.assertRaises(ValueError):
            monitor.freeze()

    def test_two_moderate_misses_required(self):
        monitor = self.monitor()
        self.assertEqual(monitor.observe(72, [295.003, 295], [294.999]*2, [295.001]*2)['status'], 'HEALTHY')
        self.assertEqual(monitor.observe(73, [295.003, 295], [294.999]*2, [295.001]*2)['status'], 'UNTRUSTED')

    def test_future_mutation_preserves_earlier_trace_and_thresholds(self):
        grid = fixture(room_rows=6, room_columns=5)
        basis = system_basis(grid, 8, adjoint=True)[0]
        obs, _, commands = generate_mismatch(grid, 'late_air_pulse', 112, Mismatch('nominal'))
        first = run_monitor(grid, basis, obs, commands)
        changed = obs.copy()
        changed[98:] += 10
        second = run_monitor(grid, basis, changed, commands)
        self.assertEqual(first['calibration_limit_K'], second['calibration_limit_K'])
        self.assertEqual([r for r in first['trace'] if r['observation'] < 98],
                         [r for r in second['trace'] if r['observation'] < 98])
        self.assertEqual(second['first_alarm'], 98)
        self.assertFalse(second['control_permission'])

    def test_bad_calibration_is_rejected(self):
        monitor = AdequacyMonitor()
        with self.assertRaises(ValueError):
            monitor.calibrate([np.nan], [0], [1])
        with self.assertRaises(ValueError):
            monitor.calibrate([0], [1], [-1])


if __name__ == '__main__':
    unittest.main()
