# -*- coding: utf-8 -*-
"""Dual-maximum VAV output map (research item 1) + 0.85-gain control.

G36 dual-maximum staging, emulated open-loop: on heating demand the reheat
valve ramps first at minimum airflow, and the damper opens only above the
split point (default 50% loop). Same heat with minimum airflow at max
discharge temperature cuts ventilation load ~1:1.

Knobs: $DM_ON (default 1; 0 = coupled map, occ7-exact), $DM_SPLIT (0.5),
$DM_HEAT_MIN (0.1). Cooling branch unchanged.
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class DualMaxOutputMixin:
    """Override of ZonePIPolicy._zone_output_map with dual-max staging."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._dm_on = int(_os.environ.get("DM_ON", "1"))
        self._dm_split = float(_os.environ.get("DM_SPLIT", "0.5"))
        self._dm_heat_min = float(_os.environ.get("DM_HEAT_MIN", "0.1"))

    def _zone_output_map(self, u):
        if not self._dm_on:
            return super()._zone_output_map(u)
        u = float(u)
        if u >= 0.0:
            # Cooling: unchanged (damper follows demand, no reheat).
            return super()._zone_output_map(u)
        h = -u
        s = self._dm_split
        valve = min(1.0, h / s)
        air = (h - s) / (1.0 - s) if h > s else 0.0
        air = max(0.0, min(1.0, air))
        damper = self._dm_heat_min + (1.0 - self._dm_heat_min) * air
        return float(damper), float(valve)


class OccDualMax7Policy(DualMaxOutputMixin, OccGated7Policy):
    """occ_gated7 with dual-maximum damper/reheat decoupling."""

    KEY = "occ_dualmax7"


class OccHDamp085G7Policy(OccGated7Policy):
    """occ_gated7 with HEAT_DAMPER_GAIN 0.85 (control arm for dual-max)."""

    KEY = "occ_hdamp085_7"
    HEAT_DAMPER_GAIN = 0.85
