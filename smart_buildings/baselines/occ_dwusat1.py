# -*- coding: utf-8 -*-
"""DWU dwusat 11x2 SAT-search port on occ9 (plan-22 P22-3).

danielwuu 0.01061 (LB #149) vs occ9 0.01236: gap 0.00175 = 8.75x floor,
rule-side. Round-5 angle-2 verdict: the joint per-AHU 11 SAT x 2 recirc
cost-min search (model-inverted min flows + coil-saturation cap +
band-edge riding) most likely explains the bulk of the gap. Online
adaptation is a residual disturbance observer (NO RLS: covariance is
alloc-only in the sha-verified archive -- parent-verified).

Phase 1 (this file): port _free_drift + residual observer + slow EMA +
_band + _choose_ahu core; recompute per-AHU SAT/recirc/pressure
post-chain, KEEP occ9 dampers/reheat/occ-gate/water/run-bits.
Phase 2 (only if phase 1 is live but sub-floor): faithful OFF candidate
(drift + cost rule).
"""
import math
import re
from collections import deque

import numpy as np

from smart_buildings.baselines.occ_econdbhhot9 import (
    OccEconDbhHot9Policy,
)


class DwusatMixin:
    """danielwuu cpu-b search core, adapted to our obs/meta."""

    DWU_DEFAULTS = {
        "comfort_margin": 0.12,
        "precondition_hours": 1.0,
        "recovery_steps": 1.5,
        "comfort_cost": 65000.0,
        "gas_cost_ratio": 0.30,
        "standby_minimum": 282.0,
        "standby_maximum": 307.0,
        "observation_delay": 1,
        "response_scale": 0.75,
        "disturbance_rate": 0.35,
    }
    DWU_AIR_CAPACITY = 1.2 * 1006.0
    DWU_WATER_VALVE_CAPACITY = 4180.0 * 1000.0 * 0.108 / 3600.0
    DWU_DESIGN_PRESSURE = 2000.0
    DWU_MIN_PRESSURE = 200.0
    DWU_MIN_DAMPER = 0.10
    DWU_MAX_RECIRCULATION = 0.70
    DWU_MAX_DISCHARGE_K = 308.15

    def _dwu_init(self, building_meta_data) -> None:
        names = list(self.field_names)
        idx = {n: i for i, n in enumerate(names)}
        # Zone order follows AHU controllers; groups index into it.
        self._dwu_zone_ids = []
        self._dwu_zone_obs = []
        self._dwu_groups = []
        for ctrl in self.ahu_controllers:
            g = []
            for vav_id, obs_idx, _ in ctrl["zones"]:
                self._dwu_zone_ids.append(vav_id)
                self._dwu_zone_obs.append(obs_idx)
                g.append(len(self._dwu_zone_ids) - 1)
            self._dwu_groups.append(np.array(g, dtype=int))
        self._dwu_n = len(self._dwu_zone_ids)
        self._dwu_oat = []
        for ctrl in self.ahu_controllers:
            self._dwu_oat.append(ctrl.get("oat_idx"))
        self._dwu_flow_obs = np.array(
            [idx.get(f"{z}_supply_air_flowrate_setpoint", -1)
             for z in self._dwu_zone_ids], dtype=int)
        self._dwu_lo_obs = np.array(
            [idx.get(f"{z}_zone_air_heating_temperature_setpoint_sensor", -1)
             for z in self._dwu_zone_ids], dtype=int)
        self._dwu_hi_obs = np.array(
            [idx.get(f"{z}_zone_air_cooling_temperature_setpoint_sensor", -1)
             for z in self._dwu_zone_ids], dtype=int)
        self._dwu_water = next(
            (i for i, n in enumerate(names)
             if n.endswith("supply_water_temperature_sensor")), None)
        self._dwu_areas, self._dwu_ua = self._dwu_geometry(
            building_meta_data.get("zone_map"))
        cap = (3800.0 * self._dwu_areas
               + 21000.0 * np.sqrt(self._dwu_areas))
        self._dwu_gain = np.clip(300000.0 / cap, 0.08, 2.5)
        self._dwu_reset()

    @staticmethod
    def _dwu_natural_key(value: str):
        return tuple(int(x) if x.isdigit() else x
                     for x in re.split(r"(\d+)", str(value)))

    def _dwu_geometry(self, zone_map):
        areas = np.full(self._dwu_n, 55.0)
        ua = np.full(self._dwu_n, 18.0)
        if zone_map is None:
            return areas, ua
        grid = np.asarray(zone_map)
        if grid.ndim != 2 or grid.size > 5000000:
            return areas, ua
        height, width = grid.shape
        inside = (grid == 0).ravel()
        visited = np.zeros(grid.size, dtype=bool)
        flat = grid.ravel()
        counts, edges = [], []
        for start in np.flatnonzero(inside):
            if visited[start]:
                continue
            stack = [int(start)]
            visited[start] = True
            count, edge = 0, 0
            while stack:
                p = stack.pop()
                count += 1
                row, col = divmod(p, width)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    q = nr * width + nc
                    if inside[q] and not visited[q]:
                        visited[q] = True
                        stack.append(q)
                    elif flat[q] == 2:
                        edge += 1
                    elif flat[q] == 1:
                        rr, cc = row + 2 * dr, col + 2 * dc
                        if (0 <= rr < height and 0 <= cc < width
                                and grid[rr, cc] == 2):
                            edge += 1
            counts.append(count)
            edges.append(edge)
        if len(counts) != self._dwu_n:
            return areas, ua
        order = sorted(range(self._dwu_n),
                       key=lambda j: self._dwu_natural_key(
                           self._dwu_zone_ids[j]))
        for rank, j in enumerate(order):
            if 0 <= rank < len(counts):
                areas[j] = max(1.0, 0.04 * counts[rank])
                ua[j] = max(0.5, edges[rank] * 0.2 * 3.0 * 1.3)
        return areas, ua

    def _dwu_reset(self) -> None:
        rs = self.DWU_DEFAULTS["response_scale"]
        self._dwu_step = 0
        self._dwu_hist = deque(maxlen=3)
        self._dwu_prev = None
        self._dwu_slow = None
        self._dwu_res = np.zeros(self._dwu_n)
        self._dwu_merr = np.full(self._dwu_n, 0.08)
        th = np.zeros((self._dwu_n, 6))
        th[:, 0] = self._dwu_gain * self._dwu_areas * 2.0 / 1000.0
        th[:, 1] = self._dwu_gain * self._dwu_ua * 20.0 / 1000.0
        th[:, 2] = 0.18
        th[:, 3] = 0.08
        th[:, 4] = self._dwu_gain * self._dwu_areas * 8.0 / 1000.0
        th[:, 5] = self._dwu_gain * rs
        th[:, :2] *= rs
        th[:, 4] *= rs
        self._dwu_theta = th

    def _dwu_group_temp(self, temperature):
        means = np.zeros(self._dwu_n)
        for g in self._dwu_groups:
            if g.size:
                means[g] = np.average(temperature[g],
                                      weights=self._dwu_areas[g])
        return means

    def _dwu_features(self, temperature, outside, mode, slow):
        return np.column_stack((
            np.ones(self._dwu_n),
            np.clip((outside - temperature) / 20.0, -5.0, 5.0),
            np.clip((slow - temperature) / 5.0, -5.0, 5.0),
            np.clip((self._dwu_group_temp(temperature) - temperature)
                      / 5.0, -5.0, 5.0),
            np.full(self._dwu_n, float(mode)),
        ))

    def _dwu_learn(self, observed) -> None:
        delay = int(self.DWU_DEFAULTS["observation_delay"])
        if self._dwu_prev is None or len(self._dwu_hist) <= delay:
            return
        applied = list(self._dwu_hist)[-(delay + 1)]
        old = self._dwu_prev
        power = (applied["conductance"] * (applied["supply"] - observed)
                 + applied["reheat"]) / 1000.0
        feat = self._dwu_features(old, applied["outside"], applied["mode"],
                                  applied["slow"])
        x = np.column_stack((feat, power))
        delta = np.clip(observed - old, -15.0, 15.0)
        pred = np.einsum("ij,ij->i", self._dwu_theta, x)
        rem = delta - pred
        rate = float(self.DWU_DEFAULTS["disturbance_rate"])
        self._dwu_merr = (0.94 * self._dwu_merr
                          + 0.06 * np.minimum(
                              np.abs(rem - self._dwu_res), 2.0))
        self._dwu_res = np.clip((1.0 - rate) * self._dwu_res
                                + rate * rem, -5.0, 5.0)

    def _dwu_drift(self, temperature, outside, mode):
        feat = self._dwu_features(temperature, outside, mode,
                                  self._dwu_slow)
        return np.clip(np.einsum("ij,ij->i", self._dwu_theta[:, :5],
                                 feat) + self._dwu_res, -5.0, 5.0)

    def _dwu_band(self, obs, temperature):
        now = self._comfort_now(obs)
        soon = self._comfort_soon(obs)
        default = (self.comfort_temp_window if now
                   else self.eco_temp_window)
        lo = np.full(self._dwu_n, default[0])
        hi = np.full(self._dwu_n, default[1])
        valid = (self._dwu_lo_obs >= 0) & (self._dwu_hi_obs >= 0)
        if np.any(valid):
            slo = obs[self._dwu_lo_obs[valid]].astype(float)
            shi = obs[self._dwu_hi_obs[valid]].astype(float)
            good = ((slo > 150.0) & (shi < 450.0) & (shi > slo))
            ids = np.flatnonzero(valid)[good]
            lo[ids] = slo[good]
            hi[ids] = shi[good]
        if now:
            lo = np.maximum(lo, self.comfort_temp_window[0])
            hi = np.minimum(hi, self.comfort_temp_window[1])
        elif soon:
            hour, _ = self._get_hour_and_day(obs)
            left = max(0.0, (self.morning_start_hour - hour) % 24.0)
            pre = self.DWU_DEFAULTS["precondition_hours"]
            prog = np.clip(1.0 - left / max(0.1, pre), 0.0, 1.0)
            lo = (self.eco_temp_window[0]
                  + prog * (self.comfort_temp_window[0]
                            - self.eco_temp_window[0])
                  + np.zeros(self._dwu_n))
            hi = (self.eco_temp_window[1]
                  + prog * (self.comfort_temp_window[1]
                            - self.eco_temp_window[1])
                  + np.zeros(self._dwu_n))
        else:
            lo[:] = self.DWU_DEFAULTS["standby_minimum"]
            hi[:] = self.DWU_DEFAULTS["standby_maximum"]
        margin = (self.DWU_DEFAULTS["comfort_margin"]
                  if (now or soon) else 0.0)
        lo += margin
        hi -= margin
        rec = max(1.0, 1.5)
        lo = np.minimum(lo, temperature
                        + np.maximum((lo - temperature) / rec, 0.0))
        hi = np.maximum(hi, temperature
                        - np.maximum((temperature - hi) / rec, 0.0))
        return lo, hi, now

    def _dwu_native(self, value, low, high):
        return np.clip(-1.0 + (np.asarray(value, dtype=float) - low)
                       * 2.0 / max(high - low, 1e-12), -1.0, 1.0)

    def _dwu_search(self, group, temperature, free, lo, hi, max_flow,
                    outside, water):
        t, d = temperature[group], free[group]
        glo, ghi = lo[group], hi[group]
        resp = self._dwu_theta[group, 5]
        cap = max_flow[group]
        inside = float(np.average(t, weights=self._dwu_areas[group]))
        targets = np.array([285.0, 287.0, 289.0, 291.0, 293.0, 295.0,
                            297.0, 299.0, 302.0, 305.0])
        targets = np.concatenate(
            (targets, [float(np.clip(np.min(ghi) - 1.5, 285.0, 305.0))]))
        setpoint = np.repeat(targets, 2)
        recirc = np.full_like(setpoint, self.DWU_MAX_RECIRCULATION)
        if outside < inside - 0.5:
            recirc[1::2] = np.clip(
                (targets - outside) / max(inside - outside, 1e-6),
                0.0, self.DWU_MAX_RECIRCULATION)
        mixed = recirc * inside + (1.0 - recirc) * outside
        supply = np.minimum(mixed, setpoint)
        sa = supply[:, None]
        ac = self.DWU_AIR_CAPACITY
        cool_need = np.maximum(t + d - ghi, 0.0) * 1000.0 / resp
        heat_need = np.maximum(glo - t - d, 0.0) * 1000.0 / resp
        cool_flow = (cool_need[None, :]
                     / (ac * np.maximum(ghi[None, :] - sa, 0.05)))
        heat_flow = (heat_need[None, :]
                     / (ac * np.maximum(self.DWU_MAX_DISCHARGE_K
                                         - glo[None, :], 0.5)))
        wvc = self.DWU_WATER_VALVE_CAPACITY
        max_water = (wvc * math.sqrt(20.0)
                     * np.maximum(water - sa, 0.0))
        max_heat_flow = max_water / (
            ac * np.maximum(self.DWU_MAX_DISCHARGE_K - sa, 0.1))
        heat_flow = np.minimum(heat_flow, max_heat_flow)
        need_flow = np.maximum(cool_flow, heat_flow)
        frac = np.max(need_flow / cap[None, :], axis=1)
        dp = self.DWU_DESIGN_PRESSURE
        pressure = np.clip(dp * frac ** 2, self.DWU_MIN_PRESSURE, dp)
        ratio = np.sqrt(pressure / dp)
        damp = np.clip(need_flow / (cap[None, :] * ratio[:, None]),
                       self.DWU_MIN_DAMPER, 1.0)
        flow = damp * cap[None, :] * ratio[:, None]
        cond = flow * ac
        wgap = np.maximum(water - sa, 0.0)
        wanted = np.maximum(
            (glo[None, :] - t[None, :] - d[None, :]) * 1000.0
            / resp[None, :] + cond * (glo[None, :] - sa), 0.0)
        lim = np.minimum(cond * np.maximum(self.DWU_MAX_DISCHARGE_K - sa,
                                           0.0),
                         wvc * math.sqrt(20.0) * wgap)
        reheat = np.minimum(wanted, lim)
        pred = ((t[None, :] + d[None, :]
                 + resp[None, :] * (cond * sa + reheat) / 1000.0)
                / (1.0 + resp[None, :] * cond / 1000.0))
        viol = (np.maximum(glo[None, :] - pred, 0.0)
                + np.maximum(pred - ghi[None, :], 0.0))
        tpen = np.sum(viol + 2.0 * viol ** 2, axis=1)
        tflow = flow.sum(axis=1)
        fan = tflow * pressure * (2.0 - recirc) / 0.9
        chil = (tflow * ac * np.maximum(mixed - supply, 0.0) / 3.5)
        gas = reheat.sum(axis=1)
        obj = (fan + chil
               + self.DWU_DEFAULTS["gas_cost_ratio"] * gas
               + self.DWU_DEFAULTS["comfort_cost"] * tpen)
        best = int(np.argmin(obj))
        return {"setpoint": float(setpoint[best]),
                "recirc": float(recirc[best]),
                "pressure": float(pressure[best]),
                "conductance": cond[best],
                "supply": float(supply[best]),
                "reheat": reheat[best]}

    def _dwu_step_search(self, obs, temperature, outside_ahu, water,
                         max_flow, mode_now):
        if self._dwu_slow is None:
            self._dwu_slow = temperature.copy()
        self._dwu_learn(temperature)
        outside = outside_ahu[
            np.array([i for i, g in enumerate(self._dwu_groups)
                      for _ in g])]
        temp = temperature.copy()
        if self._dwu_hist and int(
                self.DWU_DEFAULTS["observation_delay"]) == 1:
            last = self._dwu_hist[-1]
            drift = self._dwu_drift(temperature, last["outside"],
                                    last["mode"])
            gain = self._dwu_theta[:, 5] / 1000.0
            temp = ((temperature + drift
                     + gain * (last["conductance"] * last["supply"]
                               + last["reheat"]))
                    / (1.0 + gain * last["conductance"]))
            temp = np.clip(temp, temperature - 8.0, temperature + 8.0)
        lo, hi, now = self._dwu_band(obs, temp)
        free = self._dwu_drift(temp, outside, now)
        out = []
        for i, g in enumerate(self._dwu_groups):
            if not g.size:
                out.append(None)
                continue
            out.append(self._dwu_search(g, temp, free, lo, hi,
                                        max_flow,
                                        float(outside_ahu[i]), water))
        return out, temp, outside, now

    def _dwu_record(self, temperature, outside, now, per_ahu):
        cond = np.zeros(self._dwu_n)
        sup = np.zeros(self._dwu_n)
        reh = np.zeros(self._dwu_n)
        for g, ch in zip(self._dwu_groups, per_ahu):
            if ch is None:
                continue
            cond[g] = ch["conductance"]
            sup[g] = ch["supply"]
            reh[g] = ch["reheat"]
        self._dwu_hist.append({"conductance": cond, "supply": sup,
                               "reheat": reh, "outside": outside.copy(),
                               "mode": now,
                               "slow": self._dwu_slow.copy()})
        self._dwu_slow += ((temperature - self._dwu_slow) / 48.0)
        self._dwu_prev = temperature.copy()
        self._dwu_step += 1


class DwusatProbe(DwusatMixin, OccEconDbhHot9Policy):
    """P22-3 phase 1: occ9 chain + dwusat SAT/recirc/pressure search."""

    KEY = "occ_dwusat1"
    DWUSAT_ON = True

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._dwu_init(building_meta_data)

    def reset(self):
        super().reset()
        # Chain __init__ resets before _dwu_init runs; _dwu_init ends
        # with a reset itself, so skip until state exists.
        if getattr(self, "_dwu_n", None) is not None:
            self._dwu_reset()

    def get_action(self, obs):
        from smart_buildings.baselines.zone_pi import _outside_kelvin
        action = super().get_action(obs)
        if not self.DWUSAT_ON:
            return action
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        temp = np.array([self._denormalize_zone_temp(obs[j])
                         for j in self._dwu_zone_obs], dtype=float)
        oat = np.array([
            _outside_kelvin(self, float(obs[i])) if i is not None
            else 290.0 for i in self._dwu_oat], dtype=float)
        if self._dwu_water is not None:
            water = _outside_kelvin(self, float(obs[self._dwu_water]))
        else:
            water = 340.0
        max_flow = np.full(self._dwu_n, 2.0)
        valid = self._dwu_flow_obs >= 0
        if np.any(valid):
            rd = obs[self._dwu_flow_obs[valid]]
            max_flow[valid] = np.where(np.isfinite(rd) & (rd > 0.0),
                                       rd, 2.0)
        max_flow = np.clip(max_flow, 0.01, 100.0)
        per_ahu, tnow, outside, now = self._dwu_step_search(
            obs, temp, oat, water, max_flow, self._comfort_now(obs))
        for i, ctrl in enumerate(self.ahu_controllers):
            if per_ahu[i] is None:
                continue
            if not disc[ctrl["run_discrete_idx"]]:
                continue
            ch = per_ahu[i]
            cont[ctrl["temp_idx"]] = np.float32(self._dwu_native(
                ch["setpoint"], 285.0, 305.0))
            if ctrl["pressure_idx"] is not None:
                cont[ctrl["pressure_idx"]] = np.float32(
                    self._dwu_native(ch["pressure"], 0.0, 2000.0))
            if ctrl["recirc_idx"] is not None:
                cont[ctrl["recirc_idx"]] = np.float32(self._dwu_native(
                    ch["recirc"], 0.0, 1.0))
        self._dwu_record(tnow, outside, now, per_ahu)
        return action
