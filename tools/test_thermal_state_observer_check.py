"""Causal delayed state estimation: reset, timing and no hidden-state leakage."""

import unittest
from unittest.mock import patch

import numpy as np

from tools.thermal_grid_reference import fixture
from tools.thermal_system_modes_check import system_basis
from tools.thermal_wall_modes_check import ModalResponse
from tools.thermal_state_observer_check import DelayedObserver, check, generate_data


class ThermalStateObserverCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = fixture(room_rows=6, room_columns=5)
        cls.model = ModalResponse(cls.grid, system_basis(cls.grid, 8, adjoint=True)[0])

    def test_reset_duplicate_is_not_a_second_update(self):
        observer = DelayedObserver(self.grid, self.model)
        observer.receive([294., 296.])
        state, covariance = observer.state.copy(), observer.p.copy()
        observer.receive([294., 296.])
        np.testing.assert_array_equal(observer.state, state)
        np.testing.assert_array_equal(observer.p, covariance)
        with self.assertRaises(ValueError):
            observer.receive([294., 296.])  # missing completed u0

    def test_new_action_does_not_replace_pending_transition(self):
        observer = DelayedObserver(self.grid, self.model)
        observer.receive([295., 295.])
        observer.receive([295., 295.])
        pending = dict(ambient=295., fixed=(0., 0.), conductance=(4., 3.), supply=(280., 290.))
        warm = dict(ambient=295., fixed=(30., 0.), conductance=(4., 3.), supply=(300., 300.))
        before = observer.state.copy()
        a = observer.forecast([pending, pending])
        b = observer.forecast([pending, warm])
        np.testing.assert_array_equal(a[0], b[0])
        self.assertGreater(np.max(np.abs(a[1] - b[1])), .01)
        np.testing.assert_array_equal(observer.state, before)

    def test_future_observations_and_inputs_do_not_change_past_states(self):
        obs, inputs = generate_data(self.grid, 'mixed', 24)
        changed_obs = obs.copy()
        changed_obs[12:] += 10
        changed_inputs = [dict(values) for values in inputs]
        changed_inputs[12]['fixed'] = (10000., 0.)
        traces = []
        for data, commands in ((obs, inputs), (changed_obs, changed_inputs)):
            observer = DelayedObserver(self.grid, self.model)
            trace = []
            for t in range(12):
                trace.append(observer.receive(data[t], completed_input=commands[t - 2] if t >= 2 else None))
            traces.append(trace)
        np.testing.assert_array_equal(traces[0], traces[1])

    def test_open_loop_and_filter_have_identical_causal_initialization(self):
        corrected = DelayedObserver(self.grid, self.model)
        open_loop = DelayedObserver(self.grid, self.model, correction=False)
        with patch('tools.thermal_state_observer_check.initial_state', side_effect=AssertionError('hidden init')):
            a = corrected.receive([294., 296.])
            b = open_loop.receive([294., 296.])
        np.testing.assert_array_equal(a, b)
        self.assertLess(np.max(np.abs(self.model.s @ a - [294., 296.])), 1e-5)

    def test_generated_sensor_delay_is_exact(self):
        obs, commands = generate_data(self.grid, 'air_pulse', 12)
        np.testing.assert_array_equal(obs[0], obs[1])
        reference = ModalResponse(self.grid, np.eye(len(self.grid.capacity)))
        state = np.full(len(self.grid.capacity), 295.)
        state = reference.advance(state, commands[0])
        np.testing.assert_allclose(obs[2], self.grid.sensed.T @ state)

    def test_late_pulse_occurs_after_prefix_and_obeys_sensor_delay(self):
        obs, commands = generate_data(self.grid, 'late_air_pulse', 112)
        self.assertEqual(commands[95]['conductance'], (0., 0.))
        self.assertEqual(commands[96]['conductance'], (4., 0.))
        np.testing.assert_allclose(obs[97], [295., 295.], atol=1e-9)
        self.assertGreater(np.max(np.abs(obs[98] - obs[97])), .01)

    def test_budget_and_full_horizon_rejected_before_generation(self):
        with patch('tools.thermal_state_observer_check.fixture') as mocked:
            for kwargs in (dict(steps=289), dict(steps=10, prefix=9), dict(prefix=-1)):
                with self.assertRaises(ValueError):
                    check(**kwargs)
            mocked.assert_not_called()

    def test_short_check_has_matched_samples_and_explicit_conditional_protocol(self):
        result = check(steps=24, prefix=6)
        self.assertIn('conditional', result['protocol'])
        self.assertFalse(result['duplicate_reset_measurement_updated_twice'])
        for case in result['cases']:
            samples = set()
            for groups in case['metrics'].values():
                samples.update(metrics['samples'] for metrics in groups.values())
            self.assertEqual(len(samples), 1)


if __name__ == '__main__':
    unittest.main()
