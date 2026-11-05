# -*- coding: utf-8 -*-
"""DMAX dual-max staging stacked into occ9 (plan-22 P22-4).

fb1 shows 45.6% of heating zone-steps with damper>0.3 & valve<0.7
(round-5 angle-4): coupled-PI high-airflow heating waste. G36
dual-maximum stages reheat-first at minimum airflow, opening the
damper only above the split point. Fixed constants (no env vars).
"""
from smart_buildings.baselines.occ_dualmax import DualMaxOutputMixin
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)


class OccDmax9Policy(DualMaxOutputMixin, OccEconDbhHot9Policy):
    """P22-4: occ9 with dual-maximum damper/reheat decoupling."""

    KEY = "occ_dmax9"
    DM_SPLIT = 0.5
    DM_HEAT_MIN = 0.1

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._dm_on = 1
        self._dm_split = float(self.DM_SPLIT)
        self._dm_heat_min = float(self.DM_HEAT_MIN)
