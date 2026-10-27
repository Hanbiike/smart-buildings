# -*- coding: utf-8 -*-
"""Saturation restraint ported onto occ_gated7 (research item 4).

When full heat has been applied for SC_STALL steps without SC_RISE_K mean
recovery while OAT < SC_ARM_OAT_K, the heat plant is saturated: extra
airflow only feeds the mandatory outside-air stream, so heating-zone
dampers are capped at SC_CAP. Releases on recovery or demand drop.
(Mechanism from OccSatPolicy, re-based from round-1 onto occ_gated7:
the r1-based occ_sat/03/07 confound the cap with missing r2/r3/r4 fixes.)

Knobs: $SC_ON (default 1; 0 = occ7-exact), $SC_CAP (0.5),
$SC_ARM_OAT (263.15), $SC_STALL (24), $SC_RISE (0.5).
"""

import numpy as np

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class SatCapMixin:
    """Plant-saturation damper cap for OccGated7Policy."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._sc_on = int(_os.environ.get("SC_ON", "1"))
        self._sc_cap = float(_os.environ.get("SC_CAP", "0.5"))
        self._sc_arm_oat = float(_os.environ.get("SC_ARM_OAT", "263.15"))
        self._sc_stall = int(_os.environ.get("SC_STALL", "24"))
        self._sc_rise = float(_os.environ.get("SC_RISE", "0.5"))
        self._stall_steps = 0
        self._stall_anchor = None
        self._saturated = False

    def _track_saturation(self, obs, mean_now: float) -> None:
        armed = False
        if self._oat_any is not None:
            try:
                armed = (_outside_kelvin(self, float(obs[self._oat_any]))
                         < self._sc_arm_oat)
            except (TypeError, ValueError):
                armed = False
        heat_on = getattr(self, "_max_heat", 0.0) > 0.5
        if armed and heat_on and not self._shutdown and self._floor_mode is None:
            if self._stall_anchor is None:
                self._stall_anchor = mean_now
                self._stall_steps = 0
            self._stall_steps += 1
            if mean_now - self._stall_anchor >= self._sc_rise:
                self._stall_anchor = mean_now
                self._stall_steps = 0
                self._saturated = False
            elif self._stall_steps >= self._sc_stall:
                self._saturated = True
        else:
            self._stall_anchor = None
            self._stall_steps = 0
            self._saturated = False

    def _command_zones(self, continuous, zones, obs, low, high):
        super()._command_zones(continuous, zones, obs, low, high)
        if not self._sc_on or not self._saturated:
            return
        for vav_id, _, action_indices in zones:
            if not action_indices:
                continue
            if self._zone_u.get(vav_id, 0.0) >= 0:
                continue
            damper_idx = action_indices.get("supply_air_damper_percentage_command")
            if damper_idx is None:
                continue
            damper = (float(continuous[damper_idx]) + 1.0) * 0.5
            if damper > self._sc_cap:
                continuous[damper_idx] = np.float32(2.0 * self._sc_cap - 1.0)

    def _per_step(self, obs, mean_now: float) -> None:
        super()._per_step(obs, mean_now)
        if self._sc_on:
            self._track_saturation(obs, mean_now)


class OccSatCap7Policy(SatCapMixin, OccGated7Policy):
    """occ_gated7 with plant-saturation damper restraint."""

    KEY = "occ_satcap7"
