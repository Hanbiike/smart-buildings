"""Stable integration, prefix fitting, and delayed physical rollout invariants."""

from dataclasses import replace
import unittest

import numpy as np

from smart_buildings.baselines.equipment_balance import AIR_CAPACITY, ZoneThermalParameters
from smart_buildings.baselines.equipment_observer import EquipmentAssumptions, EquipmentObserver
from smart_buildings.baselines.zone_thermal import (
    ThermalFit, ThermalSeries, advance_zone, condition_mass, condition_mass_kalman, filtered_mass_path,
    fit_zone, forecast_path, prefix_balance_diagnostics,
)
from tools.test_equipment_observer import observer_case


def synthetic_series(count=72):
    p = ZoneThermalParameters(3e6, 60e6, 20, 3e6 / 7200, 0)
    air, mass = 295.0, 298.0
    rows = []
    for step in range(count):
        outdoor = 270 + 3 * np.sin(step / 12)
        supply = 285 if step % 12 < 6 else 303
        flow = .5 if step % 8 < 4 else .1
        reheat = 5000 if step % 16 < 8 else 0
        comfort = float(step % 36 < 24)
        following, next_mass = advance_zone(p, air, mass, outdoor=outdoor,
                                            supply=supply, flow=flow, reheat=reheat,
                                            internal=1000 * (.2 + .8 * comfort))
        rows.append((air, following, outdoor, supply, flow, reheat, comfort))
        air, mass = following, next_mass
    return ThermalSeries(*np.array(rows).T)


class ZoneThermalTest(unittest.TestCase):
    def test_uniform_temperature_and_zero_duration(self):
        p = ZoneThermalParameters(1e5, 1e6, 10, 100, 20)
        np.testing.assert_allclose(advance_zone(p, 295, 295, outdoor=295, supply=295, flow=1), [295, 295])
        self.assertEqual(advance_zone(p, 290, 300, outdoor=250, supply=285, flow=1, seconds=0), (290, 300))

    def test_stiff_internal_exchange_is_stable_and_conserves_energy(self):
        p = ZoneThermalParameters(1e5, 1e6, 0, 1e8, 0)
        air, mass = advance_zone(p, 290, 300, outdoor=250, supply=285, flow=0)
        self.assertTrue(290 <= air <= mass <= 300)
        self.assertAlmostEqual(p.air_capacity * (air - 290) + p.mass_capacity * (mass - 300), 0, places=4)

    def test_high_airflow_does_not_overshoot_supply_temperature(self):
        p = ZoneThermalParameters(1000, 1e6, 0, 0, 0)
        air, _ = advance_zone(p, 305, 305, outdoor=250, supply=285, flow=10000)
        self.assertTrue(285 <= air <= 305)

    def test_backward_euler_energy_balance(self):
        p = ZoneThermalParameters(1e5, 1e6, 20, 100, 5)
        air, mass = advance_zone(p, 295, 300, outdoor=270, supply=285, flow=.5,
                                 reheat=1000, internal=100)
        stored = p.air_capacity * (air - 295) + p.mass_capacity * (mass - 300)
        external = 300 * (20 * (270 - air) + 5 * (270 - mass)
                          + AIR_CAPACITY * .5 * (285 - air) + 1100)
        self.assertAlmostEqual(stored, external, places=4)

    def test_nonnegative_physics_and_finite_inputs_are_required(self):
        p = ZoneThermalParameters(1e5, 1e6, 20, 100, 5)
        for kwargs in [dict(flow=-1), dict(flow=0, seconds=-1), dict(flow=0, supply=np.nan)]:
            arguments = dict(outdoor=270, supply=285, flow=0)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                advance_zone(p, 295, 295, **arguments)

    def test_bounded_fit_has_small_synthetic_training_error(self):
        series = synthetic_series()
        fit = fit_zone(series)
        self.assertLess(fit.diagnostics['training_mae_K'], .03)
        self.assertLessEqual(fit.diagnostics['evaluations'], 40)
        self.assertGreater(fit.parameters.air_capacity, 0)
        self.assertFalse(fit.diagnostics['identified_all_parameters'])

    def test_one_state_ablation_has_no_mass_exchange(self):
        fit = fit_zone(synthetic_series(), two_state=False)
        self.assertEqual(fit.parameters.mass_conductance, 0)
        self.assertFalse(fit.two_state)

    def test_passive_multistep_has_explicit_prior_and_no_kalman(self):
        series = synthetic_series(24)
        fit = fit_zone(series, passive_multistep=True, max_evaluations=3)
        self.assertLessEqual(abs(fit.initial_mass - series.start[0]), 3)
        self.assertEqual(fit.diagnostics['fitting_objective'], 'passive_multistep')
        self.assertLessEqual(fit.diagnostics['evaluations'], 3)
        masses = filtered_mass_path(fit, series)
        low, high = fit.initial_mass, fit.initial_mass
        for index, mass in enumerate(masses[1:]):
            low = min(low, series.end[index])
            high = max(high, series.end[index])
            self.assertTrue(low - 1e-8 <= mass <= high + 1e-8)
        with self.assertRaises(ValueError):
            fit_zone(series, passive_multistep=True, mass_observer='kalman')

    def test_passive_balance_closes_on_exact_synthetic_physics(self):
        series = synthetic_series()
        p = ZoneThermalParameters(3e6, 60e6, 20, 3e6 / 7200, 0)
        fit = ThermalFit(p, 1000, 298, .2, True, {})
        audit = prefix_balance_diagnostics(fit, series)
        self.assertLess(audit['passive_balance_mae_W'], 1e-5)
        self.assertEqual(audit['required_mass_outside_envelope'], 0)
        wrong = replace(fit, internal_comfort_power=20000)
        self.assertGreater(prefix_balance_diagnostics(wrong, series)['passive_balance_mae_W'], 1000)

    def test_data_sensitivity_excludes_regularization_rows(self):
        fit = fit_zone(synthetic_series())
        sensitivity = fit.diagnostics['data_sensitivity']
        self.assertEqual(len(sensitivity['singular_values']), 5)
        self.assertTrue(0 <= sensitivity['smallest_to_largest_ratio'] <= 1)

    def test_matched_kalman_fit_is_bounded_and_explicit(self):
        fit = fit_zone(synthetic_series(), mass_observer='kalman')
        self.assertEqual(fit.diagnostics['mass_observer'], 'kalman')
        self.assertLessEqual(fit.diagnostics['evaluations'], 40)
        self.assertLess(fit.diagnostics['training_mae_K'], .03)
        with self.assertRaises(ValueError):
            fit_zone(synthetic_series(), mass_observer='unknown')

    def test_series_owns_arrays_and_filters_mass_causally(self):
        series = synthetic_series()
        fit = fit_zone(series.prefix(36))
        baseline = filtered_mass_path(fit, series)
        changed = series.end.copy()
        changed[50:] += 5
        mutated = replace(series, end=changed)
        shifted = filtered_mass_path(fit, mutated)
        np.testing.assert_array_equal(baseline[:51], shifted[:51])
        self.assertNotEqual(baseline[-1], shifted[-1])
        with self.assertRaises(ValueError):
            series.start[0] = 0

    def test_incomplete_training_prefix_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_zone(synthetic_series(7))

    def test_rollout_cannot_apply_new_command_to_pending_temperature(self):
        meta, obs, action = observer_case()
        observer = EquipmentObserver(meta, assumptions=EquipmentAssumptions())
        observer.observe(obs)
        observer.set_action(*action)
        observer.observe(obs)
        parameters = ZoneThermalParameters(1e6, 20e6, 20, 100, 0)
        fits = {r: ThermalFit(parameters, 1000, 295, .2, True, {}) for r in observer.rooms}
        masses = {r: 295.0 for r in observer.rooms}
        original = masses.copy()
        pending = observer.pending
        on, off = action, (action[0].copy(), np.array([0, 0]))
        a = forecast_path(observer, fits, masses, [on], 2, pending_comfort=1, current_comfort=1)
        b = forecast_path(observer, fits, masses, [off], 2, pending_comfort=1, current_comfort=1)
        self.assertEqual(a.temperatures[0], b.temperatures[0])
        self.assertNotEqual(a.temperatures[1], b.temperatures[1])
        self.assertIs(observer.pending, pending)
        self.assertEqual(masses, original)

    def test_kalman_correction_reduces_variance_and_uses_received_measurement(self):
        p = ZoneThermalParameters(1e6, 20e6, 20, 1000, 0)
        air, mass = advance_zone(p, 295, 295, outdoor=270, supply=290, flow=.5)
        corrected, variance = condition_mass_kalman(
            p, 295, 295, 4.0, air + .1, outdoor=270, supply=290, flow=.5)
        self.assertGreater(corrected, mass)
        self.assertTrue(0 <= variance < 4)
        unchanged, _ = condition_mass_kalman(p, 295, 295, 4.0, air,
                                            outdoor=270, supply=290, flow=.5)
        self.assertAlmostEqual(unchanged, mass)

    def test_disconnected_mass_cannot_be_inferred_from_air(self):
        p = ZoneThermalParameters(1e6, 20e6, 20, 0, 0)
        mass, variance = condition_mass_kalman(p, 295, 300, 4.0, 310,
                                               outdoor=270, supply=290, flow=.5)
        self.assertEqual(mass, 300)
        self.assertGreaterEqual(variance, 4)


if __name__ == '__main__':
    unittest.main()
