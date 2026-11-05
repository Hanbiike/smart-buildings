# -*- coding: utf-8 -*-
"""PRE arrival-edge prestart on occ9 (plan-22 P22-2).

Hour-6+7 comfort is -87.5/76.7% of ALL occ9 train-30 comfort (round-5
angle-4, parent-reproduced to the digit); worst steps sit at 06:00 with
~zero energy -- the plant is late. This probe doubles the prestart
margin (6 -> 12 steps, +30 min buffer) against need under-estimation.
"""
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)


class PreMargin12Policy(OccEconDbhHot9Policy):
    """P22-2: occ9 + doubled prestart margin (12 steps)."""

    KEY = "occ_prem12"
    PRESTART_MARGIN_STEPS = 12
