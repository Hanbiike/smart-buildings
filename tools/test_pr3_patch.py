# -*- coding: utf-8 -*-
"""Focused tests for the P20-3 PR#3-equivalent recirc patch."""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, "/home/a_rakhymbekov/BUILDINGS")

from smart_buildings.smart_control.simulator. \
    simulator_flexible_floor_plan import _recirc_weight


class TestRecircWeight(unittest.TestCase):
    def test_default_is_area(self):
        os.environ.pop("PR3_AIRFLOW", None)
        vav = SimpleNamespace(damper_setting=0.5,
                              max_air_flow_rate=2.0)
        self.assertEqual(_recirc_weight(vav, 123.0), 123.0)

    def test_flag_is_airflow(self):
        os.environ["PR3_AIRFLOW"] = "1"
        try:
            vav = SimpleNamespace(damper_setting=0.5,
                                  max_air_flow_rate=2.0)
            self.assertEqual(_recirc_weight(vav, 123.0), 1.0)
        finally:
            del os.environ["PR3_AIRFLOW"]

    def test_closed_damper_zero(self):
        os.environ["PR3_AIRFLOW"] = "1"
        try:
            vav = SimpleNamespace(damper_setting=0.0,
                                  max_air_flow_rate=2.0)
            self.assertEqual(_recirc_weight(vav, 123.0), 0.0)
        finally:
            del os.environ["PR3_AIRFLOW"]


if __name__ == "__main__":
    unittest.main()
