# -*- coding: utf-8 -*-
"""econ + heating-side deadband combo (T_COMBO_D distillation stack)."""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.occ_econ import OccEconMixin
from smart_buildings.baselines.occ_dbh import DeadbandHeatMixin


class OccEconDbh7Policy(OccEconMixin, DeadbandHeatMixin, OccGated7Policy):
    """occ_gated7 + explicit economizer + heating-side deadband."""

    KEY = "occ_econdbh7"
