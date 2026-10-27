# -*- coding: utf-8 -*-
"""Slot-linear controller: per-device linear heads per 5-minute time slot.

A compact, interpretable baseline on the road to RL. Every decision is a
linear map from a FIXED-dimension feature vector, so the same weights apply
to any building (shared across zones, shared across AHUs):

  zone  [Fz] -> damper, reheat
  ahu   [Fa] -> SAT, pressure, recirculation, run bit
  glob  [Fg] -> water temp, water pressure, water run bit

Time structure comes from 288 heads per bank (one per 5 minutes of the day)
x 2 banks (workday / weekend). At step N the policy blends the neighboring
heads (default 1-2-1 over N-1/N/N+1, Pascal rows for wider blends), so slot
boundaries are smooth. Bank and slot decode from the hod/dow harmonics.

`SlotFeatures` is stateful (previous + EMA zone temperature) and is used
identically in training (`tools/slot_train.py`) and deployment, so there is
no train/infer skew. Weights are plain numpy (.npz), no torch needed.
"""

import math

import numpy as np

from smart_buildings.nn_model import FeatureContext

N_SLOTS = 288
BANKS = ("workday", "weekend")


def slot_of_day(hod_sin: float, hod_cos: float) -> int:
    """5-minute slot index in [0, 288) from the fundamental hod harmonic.

    The +1e-4 absorbs float32 quantization noise in the stored harmonics
    (error ~1e-5 in slot units); a bare int() decodes grid-edge timestamps
    one slot low.
    """
    angle = math.atan2(hod_sin, hod_cos)
    if angle < 0:
        angle += 2.0 * math.pi
    frac = angle / (2.0 * math.pi)
    return min(N_SLOTS - 1, int(frac * N_SLOTS + 1e-4))


def bank_of_day(dow_sin: float, dow_cos: float) -> int:
    """0 = workday (Mon-Fri), 1 = weekend, from the dow harmonic."""
    angle = math.atan2(dow_sin, dow_cos)
    if angle < 0:
        angle += 2.0 * math.pi
    day = int(7.0 * angle / (2.0 * math.pi) + 1e-4) % 7
    return 0 if day < 5 else 1


def blend_weights(kind=(1, 2, 1)) -> np.ndarray:
    """Normalized symmetric blend kernel over neighboring slots."""
    w = np.asarray(kind, dtype=np.float64)
    return w / w.sum()


class SlotFeatures:
    """Fixed-dim features with 1-step memory. Shared train/infer path."""

    FZ_EXTRA = 10  # occupancy, comfort x2, SAT-zone, AHU run, water-zone,
    # ...prev temp, EMA temp, leaky err-integral, outdoor-air-temp
    # (appended to room_feats). The oat channel is unsaturated: room
    # feat4 (T-oat) clips at +/-2, so fairbanks days are indistinguishable
    # without it.
    FA = 14        # + heat-need EMA, cool-need EMA (teacher mixedness state)
    FG = 11        # + leaky heat-shortfall integral (teacher reset state)

    def __init__(self, meta: dict):
        self.ctx = FeatureContext(meta)
        names = self.ctx.names
        idx = self.ctx.idx
        self.hod_sin = idx["hod_sin_000"]
        self.hod_cos = idx["hod_cos_000"]
        self.dow_sin = idx["dow_sin_000"]
        self.dow_cos = idx["dow_cos_000"]
        self.occ_idx = idx.get("num_occupants", -1)
        self.now_idx = idx.get("comfort_mode_now", -1)
        self.soon_idx = idx.get("comfort_mode_soon", -1)
        self.hws_temp = idx.get("hws_supply_water_temperature_sensor", -1)
        self.hws_press = idx.get("hws_differential_pressure_sensor", -1)
        self.hws_run = idx.get("hws_run_status",
                               idx.get("hws_supervisor_run_command", -1))
        fz_room = 8
        self.FZ = fz_room + self.FZ_EXTRA
        self.reset()

    def reset(self) -> None:
        self._prev_t = None    # per-room zone temp [K]
        self._ema_t = None     # per-room EMA zone temp [K]
        self._int_z = None     # per-room leaky center-error integral
        self._heat_ema = None  # per-AHU heat-need EMA
        self._cool_ema = None  # per-AHU cool-need EMA
        self._int_w = 0.0      # building heat-shortfall integral

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _clip(v, lo=-2.0, hi=2.0):
        return np.clip(v, lo, hi)

    def _water_kelvin(self, obs) -> float:
        if self.hws_temp < 0:
            return 330.0
        v = float(obs[self.hws_temp])
        return v if v > 150.0 else v * 20.0 + 330.0

    def _oat_kelvin(self, obs) -> float:
        cols = self.ctx.oat_cols
        if not cols:
            return 280.0
        v = float(np.mean([obs[i] for i in cols]))
        entry = self.ctx.norm.get("outside_air_temperature_sensor")
        if entry is not None and abs(v) <= 150.0:
            return v * entry["sample_std"] + entry["sample_mean"]
        return v

    def _sat_kelvin(self, obs, ahu_i: int) -> float:
        cols = self.ctx.ahu_cols[ahu_i]
        i = cols["supply_air_temperature_sensor"]
        if i < 0:
            return 291.0
        v = float(obs[i])
        entry = self.ctx.norm.get("supply_air_temperature_sensor")
        if entry is not None and abs(v) <= 150.0:
            return v * entry["sample_std"] + entry["sample_mean"]
        return v

    # -- main ------------------------------------------------------------

    def step(self, obs: np.ndarray, update: bool = True):
        """Featurize one obs; returns (bank, slot, zf, af, gf).

        zf [Z, FZ], af [A, FA], gf [FG]. Memory (prev/EMA temps) reflects
        steps strictly before this one.

        update=False decodes bank/slot only and leaves memory untouched
        (features are zero dummies). Training uses it on teacher steps
        the hybrid never queries (shutdown), so memory dynamics match
        inference exactly -- the hybrid calls step() only when its
        rule chain delegates to the slot heads.
        """
        obs = np.asarray(obs, dtype=np.float64)
        ctx = self.ctx
        bank = bank_of_day(obs[self.dow_sin], obs[self.dow_cos])
        slot = slot_of_day(obs[self.hod_sin], obs[self.hod_cos])
        if not update:
            return bank, slot, (
                np.zeros((ctx.Z, self.FZ), np.float32),
                np.zeros((ctx.A, self.FA), np.float32),
                np.zeros((self.FG,), np.float32))

        room = ctx.room_feats(obs.astype(np.float32)).astype(np.float64)
        Z = ctx.Z
        occ = float(obs[self.occ_idx]) if self.occ_idx >= 0 else 0.0
        now = float(obs[self.now_idx]) if self.now_idx >= 0 else 0.0
        soon = float(obs[self.soon_idx]) if self.soon_idx >= 0 else 0.0
        tw = self._water_kelvin(obs)
        oat = self._oat_kelvin(obs)

        t_now = np.array([
            obs[r["temp_obs"]] * ctx.z_std + ctx.z_mean for r in ctx.rooms
        ])
        if self._prev_t is None:
            self._prev_t = t_now.copy()
            self._ema_t = t_now.copy()
            self._int_z = np.zeros(Z)
            self._heat_ema = np.zeros(ctx.A)
            self._cool_ema = np.zeros(ctx.A)
            self._int_w = 0.0
        center_err = (room[:, 0] - room[:, 1]) * 0.5  # (center-T)/8
        zf = np.zeros((Z, self.FZ))
        zf[:, :8] = room
        zf[:, 8] = self._clip(occ / 50.0, -1.0, 3.0)
        zf[:, 9] = now
        zf[:, 10] = soon
        for k, r in enumerate(ctx.rooms):
            sat = self._sat_kelvin(obs, r["ahu"])
            zf[k, 11] = self._clip((sat - t_now[k]) / 10.0)
            run_i = ctx.ahu_cols[r["ahu"]]["supervisor_run_command"]
            zf[k, 12] = float(obs[run_i]) if run_i >= 0 else 0.0
            zf[k, 13] = self._clip((tw - t_now[k]) / 20.0)
            zf[k, 14] = (self._prev_t[k] - 294.0) / 6.0
            zf[k, 15] = (self._ema_t[k] - 294.0) / 6.0
        zf[:, 16] = self._clip(self._int_z / 20.0)
        zf[:, 17] = self._clip((oat - 280.0) / 20.0)
        self._prev_t = t_now.copy()
        self._ema_t = (11.0 / 12.0) * self._ema_t + (1.0 / 12.0) * t_now
        self._int_z = 0.995 * self._int_z + np.clip(center_err, -1.0, 1.0)

        A = ctx.A
        af = np.zeros((A, self.FA))
        for i in range(A):
            ks = [k for k, r in enumerate(ctx.rooms) if r["ahu"] == i]
            if ks:
                rk = room[ks]
                af[i, 0] = rk[:, 0].mean()     # cool headroom
                af[i, 1] = rk[:, 1].mean()     # warmth above heat edge
                af[i, 2] = rk[:, 7].mean()     # abs temp
                af[i, 3] = (-rk[:, 1]).max()   # worst cold (positive)
                af[i, 4] = (-rk[:, 0]).max()   # worst hot (positive)
                heat_need = np.clip(-rk[:, 1] * 4.0, 0.0, 1.0).mean()
                cool_need = np.clip(-rk[:, 0] * 4.0, 0.0, 1.0).mean()
            else:
                heat_need = cool_need = 0.0
            af[i, 12] = self._heat_ema[i]
            af[i, 13] = self._cool_ema[i]
            self._heat_ema[i] += (heat_need - self._heat_ema[i]) / 12.0
            self._cool_ema[i] += (cool_need - self._cool_ema[i]) / 12.0
            af[i, 5] = self._clip((oat - 280.0) / 20.0)
            sat = self._sat_kelvin(obs, i)
            af[i, 6] = (sat - 291.0) / 10.0
            cols = ctx.ahu_cols[i]
            pi = cols["supply_air_static_pressure_sensor"]
            af[i, 7] = float(obs[pi]) / 1000.0 if pi >= 0 else 0.2
            fi = cols["supply_air_flowrate_sensor"]
            af[i, 8] = float(obs[fi]) / 4.0 if fi >= 0 else 0.0
            run_i = cols["supervisor_run_command"]
            af[i, 9] = float(obs[run_i]) if run_i >= 0 else 0.0
            rc = cols["recirculation_percentage_command"]
            af[i, 10] = float(obs[rc]) if rc >= 0 else 0.0
            cr = cols["cooling_request_count"]
            af[i, 11] = float(obs[cr]) / 4.0 if cr >= 0 else 0.0

        gf = np.zeros((self.FG,))
        gf[0] = self._clip(occ / 50.0, -1.0, 3.0)
        gf[1] = now
        gf[2] = soon
        if Z:
            gf[3] = room[:, 0].mean()
            gf[4] = room[:, 1].mean()
            gf[5] = room[:, 7].mean()
        gf[6] = (tw - 330.0) / 20.0
        gf[7] = float(obs[self.hws_press]) / 10.0 if self.hws_press >= 0 else 0.0
        gf[8] = float(obs[self.hws_run]) if self.hws_run >= 0 else 0.0
        gf[9] = self._clip((oat - 280.0) / 20.0)
        shortfall = float(np.mean(np.clip(-room[:, 1], 0.0, None))) if Z else 0.0
        gf[10] = self._clip(self._int_w / 50.0)
        self._int_w = 0.995 * self._int_w + shortfall
        return bank, slot, (np.nan_to_num(zf).astype(np.float32),
                            np.nan_to_num(af).astype(np.float32),
                            np.nan_to_num(gf).astype(np.float32))


class SlotLinearPolicy:
    """Slot-blended linear controller; weights from tools/slot_train.py."""

    KEY = "slot_linear"
    BLEND = (1, 2, 1)
    DEFAULT_WEIGHTS = "smart_buildings/results/slot_runs/slot_bc_occ4.npz"

    def __init__(self, building_meta_data, action_spec, weights_path=None):
        self.feat = SlotFeatures(building_meta_data)
        self.ctx = self.feat.ctx
        self.action_spec = action_spec
        self.n_continuous = int(building_meta_data["n_continuous_action"])
        self.n_discrete = int(building_meta_data["n_discrete_action"])
        self.blend = blend_weights(self.BLEND)
        self.half = len(self.blend) // 2
        self.Wz = self.bz = self.Wa = self.ba = self.Wg = self.bg = None
        if weights_path is None:
            weights_path = self.DEFAULT_WEIGHTS
            import os as _os
            if not _os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"slot-linear weights missing: {weights_path} "
                    "(run tools/slot_train.py first)")
        self.load(weights_path)
        # index wiring mirrors FeatureContext
        self.damp_act = [r["damp_act"] for r in self.ctx.rooms]
        self.reheat_act = [r["reheat_act"] for r in self.ctx.rooms]

    @property
    def name(self):
        return self.KEY

    def reset(self) -> None:
        self.feat.reset()

    def load(self, path) -> None:
        z = np.load(path)
        for k in ("Wz", "bz", "Wa", "ba", "Wg", "bg"):
            setattr(self, k, z[k].astype(np.float64))
        # Per-slot run thresholds: regression-to-{0,1} must be cut at the
        # slot's base rate, not 0.5, or imbalanced slots collapse to off.
        self.rate_a = (z["rate_a"].astype(np.float64) if "rate_a" in z
                       else np.full((2, N_SLOTS), 0.5))
        self.rate_w = (z["rate_w"].astype(np.float64) if "rate_w" in z
                       else np.full((2, N_SLOTS), 0.5))

    def _blended_rate(self, rate, bank, slot):
        r = 0.0
        for j, w in enumerate(self.blend):
            s = (slot + j - self.half) % N_SLOTS
            r += w * rate[bank, s]
        return float(min(max(r, 1e-6), 1.0 - 1e-6))

    def _blended(self, W, b, bank, slot):
        Wm = np.zeros_like(W[bank, slot], dtype=np.float64)
        bm = np.zeros_like(b[bank, slot], dtype=np.float64)
        for j, w in enumerate(self.blend):
            s = (slot + j - self.half) % N_SLOTS
            Wm += w * W[bank, s]
            bm += w * b[bank, s]
        return Wm, bm

    def get_action(self, obs):
        bank, slot, (zf, af, gf) = self.feat.step(np.asarray(obs))
        Wz, bz = self._blended(self.Wz, self.bz, bank, slot)
        Wa, ba = self._blended(self.Wa, self.ba, bank, slot)
        Wg, bg = self._blended(self.Wg, self.bg, bank, slot)
        zone = zf.astype(np.float64) @ Wz + bz          # [Z, 2]
        ahu = af.astype(np.float64) @ Wa + ba           # [A, 4]
        wat = gf.astype(np.float64) @ Wg + bg           # [3]

        cont = np.full(self.n_continuous, -1.0, dtype=np.float32)
        disc = np.zeros(self.n_discrete, dtype=np.int32)
        for k, (di, ri) in enumerate(zip(self.damp_act, self.reheat_act)):
            cont[di] = float(np.clip(zone[k, 0], -1.0, 1.0))
            cont[ri] = float(np.clip(zone[k, 1], -1.0, 1.0))
        thr_a = self._blended_rate(self.rate_a, bank, slot)
        for i, act in enumerate(self.ctx.ahu_actions):
            cont[act["temp_idx"]] = float(np.clip(ahu[i, 0], -1.0, 1.0))
            if act["pressure_idx"] is not None:
                cont[act["pressure_idx"]] = float(np.clip(ahu[i, 1], -1.0, 1.0))
            if act["recirc_idx"] is not None:
                cont[act["recirc_idx"]] = float(np.clip(ahu[i, 2], -1.0, 1.0))
            disc[act["run_disc_idx"]] = 1 if ahu[i, 3] > thr_a else 0
        w = self.ctx.water
        if w["water_temp_idx"] is not None:
            cont[w["water_temp_idx"]] = float(np.clip(wat[0], -1.0, 1.0))
        if w["water_pressure_idx"] is not None:
            cont[w["water_pressure_idx"]] = float(np.clip(wat[1], -1.0, 1.0))
        if 0 <= w["water_discrete_idx"] < self.n_discrete:
            thr_w = self._blended_rate(self.rate_w, bank, slot)
            disc[w["water_discrete_idx"]] = 1 if wat[2] > thr_w else 0
        cspec = self.action_spec["continuous_action"]
        dspec = self.action_spec["discrete_action"]
        cont = np.clip(cont, cspec.low, cspec.high).astype(np.float32)
        disc = np.clip(disc, dspec.low, dspec.high).astype(np.int32)
        return {"continuous_action": cont, "discrete_action": disc}
