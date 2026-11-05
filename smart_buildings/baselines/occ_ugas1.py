# -*- coding: utf-8 -*-
"""UGAS cold-unoccupied-gas trim on occ9 (plan-22 P22-1).

Cold unoccupied gas is -619.5/27.2% of the occ9 train-30 total (round-5
angle-4, parent-reproduced). weekend AHU run is 52% (floor holds).

Directionality record: FLOOR_CAPABILITY (0.5/0.7/1.0) is VACUOUS --
_Round3Mixin caps the floor at eco_lo - 1K and the cap binds whenever
the base floor is warmer, i.e. almost always (proven bit-identical on
fb8/fb1 x1500 + offline replay: floor_heat == eco_lo - 1 constantly).
The live lever is the cap itself: hold no warmer than eco_lo - 2K.
"""
from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)


class UgasCap20Policy(OccEconDbhHot9Policy):
    """P22-1: occ9 + deeper floor cap (eco_lo - 2K, less burn)."""

    KEY = "occ_ugascap20"
    FLOOR_HEAT_CAP_BELOW_ECO = 2.0
