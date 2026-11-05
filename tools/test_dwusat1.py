"""P22-3: DWUsat probe registration + search smoke."""

import unittest

import numpy as np

from smart_buildings import evaluate
from smart_buildings.baselines.occ_dwusat1 import DwusatProbe


class TestDwusat1(unittest.TestCase):
    def test_registered(self):
        self.assertIn("occ_dwusat1", evaluate.POLICY_MAP)
        self.assertIs(evaluate.POLICY_MAP["occ_dwusat1"], DwusatProbe)

    def test_flag(self):
        self.assertTrue(DwusatProbe.DWUSAT_ON)

    def test_search_smoke(self):
        probe = DwusatProbe.__new__(DwusatProbe)
        probe._dwu_areas = np.array([55.0, 55.0])
        probe._dwu_theta = np.zeros((2, 6))
        probe._dwu_theta[:, 5] = 0.5
        out = probe._dwu_search(
            group=np.array([0, 1]),
            temperature=np.array([293.0, 294.0]),
            free=np.array([0.1, -0.1]),
            lo=np.array([294.0, 294.0]),
            hi=np.array([297.0, 297.0]),
            max_flow=np.array([2.0, 2.0]),
            outside=280.0,
            water=340.0)
        self.assertIn("setpoint", out)
        self.assertGreaterEqual(out["setpoint"], 285.0)
        self.assertLessEqual(out["setpoint"], 305.0)
        self.assertGreaterEqual(out["recirc"], 0.0)
        self.assertLessEqual(out["recirc"], 0.70)
        self.assertGreaterEqual(out["pressure"], 200.0)
        self.assertLessEqual(out["pressure"], 2000.0)


if __name__ == "__main__":
    unittest.main()
