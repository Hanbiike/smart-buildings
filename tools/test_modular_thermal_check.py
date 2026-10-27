"""Prefix isolation and protocol tests without a simulator or expensive fit."""

from unittest.mock import patch
import unittest

import numpy as np

from smart_buildings.baselines.equipment_balance import ZoneThermalParameters
from smart_buildings.baselines.zone_thermal import ThermalFit
from tools.modular_thermal_check import evaluate_arrays, evaluate_case, fit_prefix
from tools.test_equipment_observer import observer_case


def small_case():
    metadata, obs, action = observer_case()
    metadata['observation_field_names'] += ['comfort_mode_now', 'num_occupants']
    observations = np.tile(np.r_[obs, 1, 2], (48, 1))
    return metadata, observations, np.tile(action[0], (48, 1)), np.tile(action[1], (48, 1))


def fixed_fit(series, *, two_state, **kwargs):
    return ThermalFit(ZoneThermalParameters(1e6, 20e6, 20, 100 if two_state else 0, 0),
                      1000, float(series.start[0]), .2, two_state,
                      dict(solver_success=True, training_transitions=len(series.start),
                           identified_all_parameters=False))


class ModularThermalCheckTest(unittest.TestCase):
    def test_invalid_budget_is_rejected_before_any_file_read(self):
        for options in [dict(steps=289), dict(steps=0), dict(offset=-1)]:
            with self.assertRaises(ValueError):
                evaluate_case('missing/case', **options)

    def test_fit_prefix_does_not_include_validation_targets(self):
        metadata, obs, continuous, discrete = small_case()
        with patch('tools.modular_thermal_check.fit_zone', side_effect=fixed_fit):
            first = evaluate_arrays(metadata, obs, continuous, discrete,
                                    fit_observations=16, horizons=(1, 3))
            changed = obs.copy()
            changed[25:, 0] += .1
            second = evaluate_arrays(metadata, changed, continuous, discrete,
                                     fit_observations=16, horizons=(1, 3))
        self.assertEqual(first['fitted_zones'], second['fitted_zones'])
        self.assertEqual(first['fitted_zones']['1']['diagnostics']['training_transitions'], 14)
        self.assertFalse(first['control_integration'])

    def test_training_uses_the_matching_delayed_control_interval(self):
        metadata, obs, continuous, discrete = small_case()
        obs[:16, -2] = np.arange(16) / 16
        supplied = []
        def capture(series, **kwargs):
            supplied.append(series)
            return fixed_fit(series, **kwargs)
        with patch('tools.modular_thermal_check.fit_zone', side_effect=capture):
            fit_prefix(metadata, obs[:16], continuous[:16], discrete[:16], two_state=True)
        np.testing.assert_array_equal(supplied[0].comfort, np.arange(14) / 16)

    def test_recorded_commands_are_explicitly_labeled_conditional(self):
        metadata, obs, continuous, discrete = small_case()
        with patch('tools.modular_thermal_check.fit_zone', side_effect=fixed_fit):
            row = evaluate_arrays(metadata, obs, continuous, discrete,
                                  fit_observations=16, horizons=(1, 3), command_path='recorded')
        self.assertTrue(row['conditional_recorded_commands'])
        self.assertTrue(row['frozen_thermal_parameters'])
        self.assertIn('occupied', row['metrics']['h3'])

    def test_unknown_protocol_is_not_silently_accepted(self):
        args = small_case()
        for kwargs in [dict(command_path='unknown'), dict(mass_observer='unknown'),
                       dict(fit_observations=48), dict(horizons=(0,))]:
            with self.assertRaises(ValueError):
                evaluate_arrays(*args, **kwargs)

    def test_kalman_protocol_reaches_prefix_fitter_and_partitions_errors(self):
        received = []
        def capture(series, **kwargs):
            received.append(kwargs['mass_observer'])
            return fixed_fit(series, **kwargs)
        with patch('tools.modular_thermal_check.fit_zone', side_effect=capture):
            row = evaluate_arrays(*small_case(), fit_observations=16,
                                  horizons=(1, 3), mass_observer='kalman')
        self.assertTrue(received)
        self.assertEqual(set(received), {'kalman'})
        self.assertTrue(row['fitting_mass_update'].startswith('kalman;'))
        for groups in row['metrics'].values():
            total = sum(groups[g]['samples'] if groups[g] else 0
                        for g in ('identified_actuators', 'uncertain_actuators'))
            self.assertEqual(total, groups['all']['samples'])

    def test_passive_multistep_protocol_and_prefix_isolation(self):
        metadata, obs, continuous, discrete = small_case()
        received = []
        def capture(series, **kwargs):
            received.append(kwargs['passive_multistep'])
            return fixed_fit(series, **kwargs)
        with patch('tools.modular_thermal_check.fit_zone', side_effect=capture):
            first = evaluate_arrays(metadata, obs, continuous, discrete,
                fit_observations=16, horizons=(1, 3), passive_multistep=True)
            changed = obs.copy()
            changed[16:, 0] += .1
            second = evaluate_arrays(metadata, changed, continuous, discrete,
                fit_observations=16, horizons=(1, 3), passive_multistep=True)
        self.assertTrue(all(received))
        self.assertEqual(first['fitted_zones'], second['fitted_zones'])
        self.assertEqual(first['mass_passive_envelope_violations'], 0)


if __name__ == '__main__':
    unittest.main()
