# -*- coding: utf-8 -*-
"""econ + dbh + heatopt stack (occ9 candidate)."""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.occ_econ import OccEconMixin
from smart_buildings.baselines.occ_dbh import DeadbandHeatMixin
from smart_buildings.baselines.occ_heatopt import HeatOptMixin


class OccEconDbhHot9Policy(HeatOptMixin, OccEconMixin, DeadbandHeatMixin,
                           OccGated7Policy):
    """occ_gated7 + economizer + heating deadband + coil-physics cap."""

    KEY = "occ_econdbhhot9"
