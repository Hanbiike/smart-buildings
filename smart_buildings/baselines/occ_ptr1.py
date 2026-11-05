# -*- coding: utf-8 -*-
"""R2 self-tuning trim-and-respond on occ9 (plan-20 P20-6).

NOT a fixed re-tune (J2 closed that door): each sub-lever adapts the
live T&R loop to per-AHU conditions. Separately gated: screen each,
promote each on its own gate.

- OccPtrA1Policy (KEY occ_ptra1): normalize PTR_PERSIST/UP by
  zones-per-AHU (big AHUs: longer persist, gentler up-steps).
- OccPtrB1Policy (KEY occ_ptrb1): G36 rogue-zone exclusion (a lone
  starved zone does not advance the starve counter on 3+ zone AHUs).
- OccPtrC1Policy (KEY occ_ptrc1): self-tune db_eff from the trailing
  per-AHU starve-request-rate EMA (frequent requests widen the
  deadband, quiet loops keep the OAT-driven default).

Architecture: PtrLoopMixin sits BETWEEN DeadbandHeatMixin and
OccGated7Policy in the occ9 base list, so ``super(OccGated7Policy,
self)._base_action`` replays the gated4-down chain (skipping only
the fixed gated7 loop) and the econ/dbh/heatopt mixins apply above
OUR loop exactly as they apply above the gated7 loop in occ9.
Ptr1Base with default hooks reproduces occ9 bit-for-bit.
"""

import numpy as np

from smart_buildings.baselines.occ_dbh import DeadbandHeatMixin
from smart_buildings.baselines.occ_econ import OccEconMixin
from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.occ_heatopt import HeatOptMixin
from smart_buildings.baselines.zone_pi import _outside_kelvin


class PtrLoopMixin:
    """Hookable T&R loop replacing the fixed OccGated7Policy loop."""

    PTR_REQ_EMA = 0.05

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._ptr_req_ema = [0.0] * len(self.ahu_controllers)

    # Hook defaults reproduce the fixed loop exactly.
    def _ptr_up_eff(self, n_zones):
        return self.PTR_UP, self.PTR_PERSIST

    def _ptr_starved_fire(self, n_zones, starve_count):
        return starve_count >= 1

    def _ptr_db_eff(self, i, base_db):
        return base_db

    def _base_action(self, obs):
        action = super(OccGated7Policy, self)._base_action(obs)
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        active = self._occupied(obs)
        lo, hi = self.comfort_temp_window
        base_db = self.PTR_DB
        if (self.PTR_DB_SLOPE > 0
                and getattr(self, "_oat_any", None) is not None):
            try:
                oat = _outside_kelvin(self, float(obs[self._oat_any]))
                dt = abs(oat - 0.5 * (lo + hi))
                base_db = (self.PTR_DB + self.PTR_DB_SLOPE
                           * max(0.0, dt - self.PTR_DB_REF))
            except (TypeError, ValueError):
                pass
        for i, ctrl in enumerate(self.ahu_controllers):
            pidx = ctrl["pressure_idx"]
            if pidx is None:
                continue
            if not disc[ctrl["run_discrete_idx"]]:
                self._ptr_p[i] = -1.0
                self._ptr_nstarve[i] = 0
                continue
            if not active:
                continue
            db_eff = self._ptr_db_eff(i, base_db)
            starve_count = 0
            all_closed = True
            for vav_id, obs_idx, _ in ctrl["zones"]:
                aidx = self.zone_action_indices.get(vav_id, {}).get(
                    "supply_air_damper_percentage_command")
                if aidx is None:
                    continue
                d = (float(cont[aidx]) + 1.0) * 0.5
                if d >= 0.5:
                    all_closed = False
                t = self._denormalize_zone_temp(obs[obs_idx])
                if d > self.PTR_DSAT and (t < lo - db_eff
                                          or t > hi + db_eff):
                    starve_count += 1
            n = sum(1 for v, _, _ in ctrl["zones"]
                    if self.zone_action_indices.get(v, {}).get(
                        "supply_air_damper_percentage_command")
                    is not None)
            starved = self._ptr_starved_fire(n, starve_count)
            ema = self._ptr_req_ema
            ema[i] += self.PTR_REQ_EMA * (float(starved) - ema[i])
            if starved:
                self._ptr_nstarve[i] += 1
            else:
                self._ptr_nstarve[i] = 0
            up_eff, persist_eff = self._ptr_up_eff(max(n, 1))
            if self._ptr_nstarve[i] >= persist_eff:
                self._ptr_p[i] = min(self.PTR_MAX,
                                     self._ptr_p[i] + up_eff)
            elif all_closed:
                self._ptr_p[i] = max(-1.0, self._ptr_p[i] - self.PTR_DOWN)
            cont[pidx] = np.float32(self._ptr_p[i])
        return action


class Ptr1Base(HeatOptMixin, OccEconMixin, DeadbandHeatMixin,
              PtrLoopMixin, OccGated7Policy):
    """occ9 stack with the hookable T&R loop (base = occ9 behavior)."""

    KEY = "occ_ptr0"


class OccPtrA1Policy(Ptr1Base):
    """R2-A: normalize persist/up-steps by zones-per-AHU."""

    KEY = "occ_ptra1"

    def _ptr_up_eff(self, n_zones):
        n = max(n_zones, 1)
        return self.PTR_UP / (n ** 0.5), self.PTR_PERSIST + (n - 1)


class OccPtrB1Policy(Ptr1Base):
    """R2-B: G36 rogue-zone exclusion on 3+ zone AHUs."""

    KEY = "occ_ptrb1"

    def _ptr_starved_fire(self, n_zones, starve_count):
        if n_zones >= 3:
            return starve_count >= 2
        return starve_count >= 1


class OccPtrC1Policy(Ptr1Base):
    """R2-C: widen db_eff with the trailing request-rate EMA."""

    KEY = "occ_ptrc1"

    def _ptr_db_eff(self, i, base_db):
        return base_db * (1.0 + self._ptr_req_ema[i])
