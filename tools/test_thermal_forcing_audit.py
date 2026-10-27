"""Oracle ablation contracts and pending-step alignment."""

import unittest
from tools.test_modular_thermal_check import small_case, fixed_fit
from tools.thermal_forcing_audit import audit
from tools.modular_thermal_check import fit_prefix
from unittest.mock import patch


class ThermalForcingAuditTest(unittest.TestCase):
    def test_oracle_and_projected_share_pending_step_and_origins(self):
        metadata, obs, continuous, discrete = small_case()
        with patch('tools.modular_thermal_check.fit_zone', side_effect=fixed_fit):
            fits = fit_prefix(metadata, obs[:16], continuous[:16], discrete[:16], two_state=True)
        result = audit(metadata, obs, continuous, discrete, fits, prefix=16, horizons=(1, 3))
        self.assertAlmostEqual(result['h1']['oracle']['model_mae_K'],
                               result['h1']['projected']['model_mae_K'])
        for groups in result.values():
            self.assertEqual(groups['oracle']['samples'], groups['projected']['samples'])
            self.assertEqual(groups['oracle']['persistence_mae_K'],
                             groups['projected']['persistence_mae_K'])
        self.assertEqual(result['h1']['oracle']['samples'], result['h3']['oracle']['samples'])

    def test_window_limit_precedes_observer_creation(self):
        with self.assertRaises(ValueError):
            audit({}, [0] * 289, [0] * 289, [0] * 289, {})


if __name__ == '__main__':
    unittest.main()
