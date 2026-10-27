# -*- coding: utf-8 -*-
"""Combined banked wins: occupancy-proportional err (item 7) + cooling SAT
shift (item 4 tail). Orthogonal hooks (_zone_err vs _zone_target)."""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.occ_prop import OccPropMixin
from smart_buildings.baselines.occ_coolshift7 import CoolShiftMixin


class OccCombo7Policy(OccPropMixin, CoolShiftMixin, OccGated7Policy):
    """occ_gated7 + prop err scaling + coolshift SAT shift."""

    KEY = "occ_combo7"
