# -*- coding: utf-8 -*-
"""Heating-side-only deadband, distilled from sb-t-combo-d.ipynb (public 0.012).

Their DSA_DeadbandHeat: zones between lo+DB and the band centre are left
alone (err = 0, integral leaks by INTEGRAL_LEAK); above the centre the PI
regulates to the centre as usual; below lo+DB it heats to lo+DB. Shipped
at DB 0.15 K with leak 0.97. This differs from our item-2 edge riding (which
floated BOTH sides to the edges and died): the cooling side still regulates
to the centre here.

Knobs: $DBH_ON (default 1; 0 = occ7-exact), $DBH_DB (default 0.15 K),
$DBH_LEAK (default 0.97).
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class DeadbandHeatMixin:
    """Heating-side deadband error + integral leak."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._dbh_on = int(_os.environ.get("DBH_ON", "1"))
        self._dbh_db = float(_os.environ.get("DBH_DB", "0.15"))
        self._dbh_leak = float(_os.environ.get("DBH_LEAK", "0.97"))

    def _zone_err(self, obs, temp, lo, hi):
        if not self._dbh_on:
            return super()._zone_err(obs, temp, lo, hi)
        temp = float(temp)
        centre = 0.5 * (lo + hi)
        if temp > centre:
            return super()._zone_err(obs, temp, lo, hi)
        if temp < lo + self._dbh_db:
            return temp - (lo + self._dbh_db)
        return 0.0

    def _zone_integral_update(self, err, integral):
        if self._dbh_on and err == 0.0:
            return integral * self._dbh_leak
        return super()._zone_integral_update(err, integral)


class OccDbh7Policy(DeadbandHeatMixin, OccGated7Policy):
    """occ_gated7 with heating-side deadband (T_COMBO_D distillation)."""

    KEY = "occ_dbh7"
