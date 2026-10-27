# -*- coding: utf-8 -*-
"""Shared observation -> feature decoding for the BC model.

`FeatureContext` is built once per building from `building_meta_data` (the
same mapping the shipped policies use) and decodes raw observations into

    room features  [Z, 8]   band deltas, zone/OAT delta, absolute temp,
                            damper/reheat commands
    ahu features   [A, 10]  OAT/SAT kelvin, SAT delta/setpoint, pressure and
                            flow ratios, fan/recirc commands, request/run
    global         [Fg]     occupant, comfort flags, hod/dow harmonics, HWS

Works on a single obs row (policy runtime) or a [T, D] matrix (tokenizer);
identical code path keeps training and deployment in sync. Scales follow the
policy decoding rules: zone-air temperature is normalized (metadata), while
zone setpoints and AHU temperature sensors arrive in raw kelvin unless the
normalized value looks like one (`<=150`), flow/pressure fields are raw.
"""
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE_K = 10.0
DIFF_K = 8.0

WINDOW = 72
STRIDE = 6
N_HIST = WINDOW // STRIDE              # 12 history slots


def _numeric_suffix(name: str) -> int:
    """Device ordering used by the shipped policies (pid._numeric_suffix)."""
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


class FeatureContext:
    def __init__(self, meta: dict):
        self.names = list(meta["observation_field_names"])
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.norm = meta["observation_normalization_metadata"]

        # zone-temp normalizer family (the policy decodes zone temps with it)
        zn = self.norm["zone_air_temperature_sensor"]
        self.z_mean, self.z_std = zn["sample_mean"], zn["sample_std"]

        # AHU wiring, identical to the shipped policies: numeric-suffix order
        # (pid._numeric_suffix), truncated to the reportable temp indices.
        ahu_keys = sorted(meta["ahu_zone_assignments"].keys(),
                          key=_numeric_suffix)
        temp_idx = meta["temp_indices"]
        n_ahu = min(len(temp_idx), len(ahu_keys))
        self.ahus = ahu_keys[:n_ahu]

        # per-room decoding indices, from AHU assignments + setpoint sensors
        self.rooms = []
        for ahu_i, ahu in enumerate(self.ahus):
            for room in meta["ahu_zone_assignments"][ahu]:
                vav = room.replace("room_", "VAV_")
                temp_field = f"{vav}_zone_air_temperature_sensor"
                if temp_field not in self.idx:
                    continue
                room_field = temp_field
                heat_field = f"{vav}_zone_air_heating_temperature_setpoint_sensor"
                cool_field = f"{vav}_zone_air_cooling_temperature_setpoint_sensor"
                act = meta["zone_action_indices"].get(vav, {})
                self.rooms.append({
                    "temp_obs": self.idx[temp_field],
                    "room_temp_obs": self.idx.get(room_field, -1),
                    "heat_obs": self.idx.get(heat_field, -1),
                    "cool_obs": self.idx.get(cool_field, -1),
                    "damp_obs": self.idx.get(
                        f"{vav}_supply_air_damper_percentage_command", -1),
                    "reheat_obs": self.idx.get(
                        f"{vav}_reheat_valve_percentage_command", -1),
                    "damp_act": act.get("supply_air_damper_percentage_command"),
                    "reheat_act": act.get("reheat_valve_percentage_command"),
                    "ahu": ahu_i,
                })
        self.rooms = [r for r in self.rooms
                      if r["damp_act"] is not None and r["reheat_act"] is not None]
        self.Z = len(self.rooms)

        # per-AHU obs columns
        self.ahu_cols = []
        for ahu in self.ahus:
            p = f"ahs_{ahu}_"
            self.ahu_cols.append({s: self.idx.get(f"{p}{s}", -1) for s in (
                "outside_air_temperature_sensor",
                "supply_air_temperature_sensor",
                "supply_air_temperature_setpoint",
                "supply_air_static_pressure_sensor",
                "supply_air_static_pressure_setpoint",
                "supply_air_flowrate_sensor",
                "supply_air_flowrate_setpoint",
                "recirculation_percentage_command",
                "supply_fan_speed_percentage_command",
                "cooling_request_count",
                "supervisor_run_command",
            )})
        self.A = len(self.ahus)

        # per-AHU action wiring, exactly pid._build_ahu_mapping + the
        # pid_plus recirculation attachment: run_disc = position, water is the
        # last pressure index beyond the AHU list.
        pressure_indices = meta["pressure_indices"]
        recirculation_indices = meta["recirculation_indices"]
        water_indices = meta["water_indices"]
        self.ahu_actions = [{
            "temp_idx": temp_idx[i],
            "pressure_idx": (pressure_indices[i]
                             if i < len(pressure_indices) else None),
            "recirc_idx": (recirculation_indices[i]
                           if i < len(recirculation_indices) else None),
            "run_disc_idx": i,
        } for i in range(self.A)]
        self.water = {
            "water_temp_idx": (water_indices[0] if water_indices else None),
            "water_pressure_idx": (pressure_indices[-1]
                                   if len(pressure_indices) > self.A else None),
            "water_discrete_idx": int(meta["n_discrete_action"]) - 1,
        }

        self.oat_cols = [i for i, nm in enumerate(self.names)
                         if nm.endswith("_outside_air_temperature_sensor")]

        self.glob_cols = [self.idx[nm] for nm in self.names
                          if nm in ("num_occupants", "comfort_mode_now",
                                    "comfort_mode_soon")
                          or nm.startswith(("hod_", "dow_", "hws_"))]
        self.fg = len(self.glob_cols)

    def _kelvin(self, v: np.ndarray, key: str) -> np.ndarray:
        entry = self.norm.get(key)
        if entry is not None and abs(float(np.mean(v))) <= 150.0:
            return v * entry["sample_std"] + entry["sample_mean"]
        return v

    def room_feats(self, obs: np.ndarray) -> np.ndarray:
        """obs [D] or [T, D] -> [Z, 8] or [T, Z, 8]."""
        td = obs.ndim == 2
        Z, F = self.Z, 8
        out_shape = (obs.shape[0], Z, F) if td else (Z, F)
        feats = np.zeros(out_shape, dtype=np.float32)
        oat_all = None
        if self.oat_cols:
            cols = obs[:, self.oat_cols] if td else obs[self.oat_cols]
            oat_all = self._kelvin(np.asarray(cols).mean(axis=-1),
                                   "outside_air_temperature_sensor")
        for k, r in enumerate(self.rooms):
            get = (lambda i: obs[:, i]) if td else (lambda i: obs[i])
            t_K = get(r["temp_obs"]) * self.z_std + self.z_mean
            sp_h = get(r["heat_obs"]) if r["heat_obs"] >= 0 else 291.0
            sp_c = get(r["cool_obs"]) if r["cool_obs"] >= 0 else 297.0
            if td:
                if np.ndim(sp_h) == 0:
                    sp_h = np.full(t_K.shape, float(sp_h))
                if np.ndim(sp_c) == 0:
                    sp_c = np.full(t_K.shape, float(sp_c))
            sp_h, sp_c = np.asarray(sp_h, dtype=np.float64), np.asarray(
                sp_c, dtype=np.float64)
            feats[..., k, 0] = np.clip((sp_c - t_K) / DIFF_K, -1.5, 1.5)
            feats[..., k, 1] = np.clip((t_K - sp_h) / DIFF_K, -1.5, 1.5)
            feats[..., k, 2] = (0.5 * (sp_h + sp_c) - 294.0) / SCALE_K
            feats[..., k, 3] = np.clip((sp_c - sp_h) / 6.0, -2.0, 2.0)
            if oat_all is not None:
                feats[..., k, 4] = np.clip((t_K - oat_all) / 12.0, -2.0, 2.0)
            else:
                feats[..., k, 4] = 0.0
            if r["damp_obs"] >= 0:
                feats[..., k, 5] = get(r["damp_obs"])
            if r["reheat_obs"] >= 0:
                feats[..., k, 6] = get(r["reheat_obs"])
            feats[..., k, 7] = (t_K - 294.0) / 6.0
        return feats

    def ahu_feats(self, obs: np.ndarray) -> np.ndarray:
        td = obs.ndim == 2
        A, F = self.A, 10
        out_shape = (obs.shape[0], A, F) if td else (A, F)
        feats = np.zeros(out_shape, dtype=np.float32)
        for i, cols in enumerate(self.ahu_cols):
            get = lambda s, r: (obs[:, r] if td else obs[r])
            oat = self._kelvin(get("oat", cols["outside_air_temperature_sensor"]),
                               "outside_air_temperature_sensor")
            sat = self._kelvin(get("sat", cols["supply_air_temperature_sensor"]),
                               "supply_air_temperature_sensor")
            satsp = self._kelvin(get("sp", cols["supply_air_temperature_setpoint"]),
                                 "supply_air_temperature_setpoint")
            feats[..., i, 0] = np.clip((oat - 280) / 20.0, -2.0, 2.0)
            feats[..., i, 1] = (sat - 291) / SCALE_K
            feats[..., i, 2] = np.clip((sat - satsp) / 5.0, -2.0, 2.0)
            feats[..., i, 3] = (satsp - 291) / SCALE_K
            pc = cols["supply_air_static_pressure_sensor"]
            pcsp = cols["supply_air_static_pressure_setpoint"]
            if pc >= 0 and pcsp >= 0:
                feats[..., i, 4] = np.clip(get("p", pc) - get("psp", pcsp),
                                           -2.0, 2.0)
            fc = cols["supply_air_flowrate_sensor"]
            fcsp = cols["supply_air_flowrate_setpoint"]
            if fc >= 0 and fcsp >= 0:
                denom = get("f", fcsp) + 1e-6
                feats[..., i, 5] = np.clip(get("f", fc) / denom - 1.0,
                                           -2.0, 2.0)
            rc = cols["recirculation_percentage_command"]
            if rc >= 0:
                feats[..., i, 6] = get("r", rc)
            fan = cols["supply_fan_speed_percentage_command"]
            if fan >= 0:
                feats[..., i, 7] = get("fn", fan)
            cr = cols["cooling_request_count"]
            if cr >= 0:
                entry = self.norm.get("cooling_request_count")
                v = get("cr", cr)
                if entry is not None and abs(float(np.mean(v))) <= 4.0:
                    v = v * entry["sample_std"] + entry["sample_mean"]
                feats[..., i, 8] = v / 4.0
            run = cols["supervisor_run_command"]
            if run >= 0:
                entry = self.norm.get("supervisor_run_command")
                v = get("s", run)
                if entry is not None and abs(float(np.mean(v))) <= 2.0:
                    v = v * entry["sample_std"] + entry["sample_mean"]
                feats[..., i, 9] = v
        return feats

    def glob_feats(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 2:
            return obs[:, self.glob_cols].astype(np.float32)
        return obs[self.glob_cols].astype(np.float32)


def hist_pool_features(zone: np.ndarray, ahu: np.ndarray, glob: np.ndarray,
                       t: int, n_hist: int = N_HIST, stride: int = STRIDE) -> np.ndarray:
    """Pooled history features before t (oldest first): [n_hist, Fh]."""
    feats = np.zeros((n_hist, zone.shape[-2] + ahu.shape[-2] + glob.shape[-1]),
                     dtype=np.float32)
    for k in range(n_hist):
        s = max(0, t - (n_hist - k) * stride)
        feats[k] = np.concatenate([zone[s].mean(axis=-2),
                                   ahu[s].mean(axis=-2), glob[s]])
    return feats


def hist_pooled_cache(zone: np.ndarray, ahu: np.ndarray, glob: np.ndarray,
                      n_hist: int = N_HIST, stride: int = STRIDE) -> np.ndarray:
    """Vectorized pooled-history for every t in [0, T): [T, n_hist, Fh]."""
    T = zone.shape[0]
    fh = zone.shape[-2] + ahu.shape[-2] + glob.shape[-1]
    pool = np.concatenate([zone.mean(axis=-2), ahu.mean(axis=-2), glob],
                          axis=-1)                       # [T, Fh]
    pooled = np.zeros((T, n_hist, fh), dtype=np.float32)
    tt = np.arange(T)
    for k in range(n_hist):
        s = np.maximum(tt - (n_hist - k) * stride, 0)
        pooled[:, k] = pool[s]
    return pooled


class EncoderBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        h = self.ln(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln(x))


class ControllerNet(nn.Module):
    """Windowed Zipformer-lite controller (see tools/bc_train.py docs)."""

    def __init__(self, fz, fa, fg, d=64, layers=2, heads=4):
        super().__init__()
        self.d = d
        self.proj_zone = nn.Linear(fz, d)
        self.proj_ahu = nn.Linear(fa, d)
        self.proj_glob = nn.Linear(fg, d)
        self.proj_hist = nn.Linear(fz + fa + fg, d)
        self.hist_pos = nn.Parameter(torch.zeros(N_HIST, d))
        self.blocks = nn.ModuleList(
            EncoderBlock(d, heads) for _ in range(layers))
        self.out_ctx = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.zone_head = nn.Linear(d, 2)                  # damper, reheat
        self.ahu_head = nn.Linear(d, 4)                   # SAT, p, recirc, run
        self.glob_head = nn.Linear(d, 3)                  # water t, p, run

    def forward(self, zf, af, gf, hist_pooled):
        """zf [B,Z,Fz] af [B,A,Fa] gf [B,Fg] hist [B,H,Fz+Fa+Fg].

        All windows in a batch share one scenario => same Z, A.
        """
        B, Z, _ = zf.shape
        A = af.shape[1]
        d = self.d
        t_zone = F.relu(self.proj_zone(zf))               # [B, Z, d]
        t_ahu = F.relu(self.proj_ahu(af))                 # [B, A, d]
        t_glob = F.relu(self.proj_glob(gf)).unsqueeze(1)  # [B, 1, d]
        t_hist = F.relu(self.proj_hist(hist_pooled)) + self.hist_pos
        x = torch.cat([t_hist, t_zone, t_ahu, t_glob], dim=1)  # [B, L, d]
        for block in self.blocks:
            x = block(x)
        n_h = N_HIST
        h_hist = x[:, :n_h]                               # [B, H, d]
        h_zone = x[:, n_h:n_h + Z]                        # [B, Z, d]
        h_ahu = x[:, n_h + Z:n_h + Z + A]                 # [B, A, d]
        h_glob = x[:, -1]                                 # [B, d]
        ctx = torch.cat([h_hist.mean(dim=1), h_zone.mean(dim=1)], dim=-1)
        ctx = self.out_ctx(ctx)                            # [B, d]
        ctx_z = ctx.unsqueeze(1) + h_zone                  # [B, Z, d]
        ctx_a = ctx.unsqueeze(1) + h_ahu                   # [B, A, d]
        ctx_g = ctx + h_glob                               # [B, d]
        return {
            "zone": self.zone_head(ctx_z),
            "ahu_sat": self.ahu_head(ctx_a)[..., 0],
            "ahu_pressure": self.ahu_head(ctx_a)[..., 1],
            "ahu_recirc": self.ahu_head(ctx_a)[..., 2],
            "ahu_run_logit": self.ahu_head(ctx_a)[..., 3],
            "water": self.glob_head(ctx_g)[..., 0],
            "water_pressure": self.glob_head(ctx_g)[..., 1],
            "water_run_logit": self.glob_head(ctx_g)[..., 2],
        }
