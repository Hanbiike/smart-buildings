# -*- coding: utf-8 -*-
"""PPO finetune of the slot-hybrid controller.

Rules (modes, run bits) stay fixed; PPO optimizes the slot-linear heads for
continuous actions (dampers, reheat, SAT, pressure, recirculation, water)
plus a slot-linear value head. On-policy training eliminates the BC
train/infer mismatch on memory features by construction.

  Rollouts: one worker process per scenario (spawn), fresh reset per
    iteration, T steps with Gaussian exploration on continuous outputs.
  Update (main process, torch CPU): clipped PPO + value MSE + KL early stop.
  Checkpoints: results/slot_runs/slot_ppo_<iter>.npz (deployment format).

Init: BC weights with memory-feature columns zeroed (stable P-only start),
small exploration std. Shutdown steps carry no gradient (no decision made)
but count for GAE/value.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from smart_buildings.baselines.slot_hybrid import SlotHybridPolicy  # noqa: E402
from smart_buildings.baselines.slot_linear import (  # noqa: E402
    N_SLOTS,
    blend_weights,
)

RUNS_DIR = PROJECT_ROOT / "smart_buildings" / "results" / "slot_runs"
SCEN_DIR = PROJECT_ROOT / "smart_buildings" / "training_scenarios"

SUBSET_8 = [
    ("building_train_1", "fairbanks_1"),
    ("building_train_1", "atlanta_2"),
    ("building_train_1", "singapore_8"),
    ("building_train_1", "phoenix_5"),
    ("building_train_8", "fairbanks_8"),
    ("building_train_8", "atlanta_1"),
    ("building_train_8", "singapore_7"),
    ("building_train_8", "phoenix_4"),
]

# Memory-feature columns (zeroed at init for a stable P-only start).
ZMEM = (16,)
AMEM = (12, 13)
WMEM = (10,)


def parse_arguments(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default=str(RUNS_DIR / "slot_bc_occ4.npz"))
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--rollout-steps", type=int, default=2016)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios", default="subset8",
                    help="'subset8', 'all', or comma list building/weather")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--kl-stop", type=float, default=0.01)
    ap.add_argument("--v-coef", type=float, default=0.5)
    ap.add_argument("--v-lr", type=float, default=1e-3)
    ap.add_argument("--v-steps", type=int, default=100)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--min-std", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="slot_ppo")
    ap.add_argument("--resume", default=None,
                    help="npz checkpoint to resume policy+value from")
    return ap.parse_args(argv)


class PPOPolicy(SlotHybridPolicy):
    """Slot-hybrid with Gaussian exploration on continuous outputs."""

    def __init__(self, building_meta_data, action_spec, weights_path,
                 logstd_z=-3.0, logstd_a=-3.0, logstd_w=-3.0):
        super().__init__(building_meta_data, action_spec, weights_path)
        self.logstd = {"z": logstd_z, "a": logstd_a, "w": logstd_w}
        self.rng = np.random.default_rng(0)
        self._valid = False
        self._record = None

    def set_noise(self, logstd_z, logstd_a, logstd_w, seed):
        self.logstd = {"z": logstd_z, "a": logstd_a, "w": logstd_w}
        self.rng = np.random.default_rng(seed)

    def get_action(self, obs):
        self._valid = False
        self._record = None
        return super().get_action(obs)

    def _chain_action(self, obs):
        # Rules chain only (skip SlotHybrid's deterministic overwrite);
        # means + noise are applied below with recording for PPO.
        action = super(SlotHybridPolicy, self)._chain_action(obs)
        bank, slot, (zf, af, gf) = self.slot.step(np.asarray(obs))
        Wz, bz = self._blended(self.Wz, self.bz, bank, slot)
        Wa, ba = self._blended(self.Wa, self.ba, bank, slot)
        Wg, bg = self._blended(self.Wg, self.bg, bank, slot)
        mz = zf.astype(np.float64) @ Wz + bz
        ma = af.astype(np.float64) @ Wa + ba
        mw = gf.astype(np.float64) @ Wg + bg
        sz = np.exp(self.logstd["z"])
        sa = np.exp(self.logstd["a"])
        sw = np.exp(self.logstd["w"])
        nz = mz + sz * self.rng.standard_normal(mz.shape)
        na = ma + sa * self.rng.standard_normal(ma.shape)
        nw = mw + sw * self.rng.standard_normal(mw.shape)
        cont = action["continuous_action"]
        rooms = self.slot.ctx.rooms
        for k, r in enumerate(rooms):
            cont[r["damp_act"]] = np.float32(np.clip(nz[k, 0], -1.0, 1.0))
            cont[r["reheat_act"]] = np.float32(np.clip(nz[k, 1], -1.0, 1.0))
        for i, act in enumerate(self.slot.ctx.ahu_actions):
            cont[act["temp_idx"]] = np.float32(np.clip(na[i, 0], -1.0, 1.0))
            if act["pressure_idx"] is not None:
                cont[act["pressure_idx"]] = np.float32(
                    np.clip(na[i, 1], -1.0, 1.0))
            if act["recirc_idx"] is not None:
                cont[act["recirc_idx"]] = np.float32(
                    np.clip(na[i, 2], -1.0, 1.0))
        w = self.slot.ctx.water
        if w["water_temp_idx"] is not None:
            cont[w["water_temp_idx"]] = np.float32(np.clip(nw[0], -1.0, 1.0))
        if w["water_pressure_idx"] is not None:
            cont[w["water_pressure_idx"]] = np.float32(
                np.clip(nw[1], -1.0, 1.0))
        self._valid = True
        self._record = {
            "bank": bank, "slot": slot,
            "zf": zf, "af": af, "gf": gf,
            "az": nz.astype(np.float32), "aa": na[:, :3].astype(np.float32),
            "aw": nw[:2].astype(np.float32),
        }
        return action


# ---------------------------------------------------------------- workers

_WORKER = {}


def _worker_init(counter, assigns):
    from smart_buildings import evaluate as E
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    scenario, gin_path = assigns[i % len(assigns)]
    env = E.load_hybrid_environment(Path(gin_path))
    env.metrics_path = None
    meta = E.create_building_meta_data(env, Path(gin_path).parent)
    spec = env.action_spec()
    tracker = E.SimulationTracker(env, capture_snapshots=False)
    _WORKER.update(scenario=scenario, env=env, meta=meta, spec=spec,
                   tracker=tracker, E=E, policy=None)


def _rollout(task):
    weights_path, logstd, steps, seed = task
    W = _WORKER
    E = W["E"]
    if W["policy"] is None:
        W["policy"] = PPOPolicy(W["meta"], W["spec"], weights_path)
    pol = W["policy"]
    z = np.load(weights_path)
    for k in ("Wz", "bz", "Wa", "ba", "Wg", "bg"):
        setattr(pol, k, z[k].astype(np.float64))
    pol.set_noise(*logstd, seed)
    pol.reset()
    tracker = W["tracker"]
    tracker.reset()
    obs = tracker.last_obs
    Z = pol.slot.ctx.Z
    A = pol.slot.ctx.A
    FZ, FA, FG = pol.slot.FZ, pol.slot.FA, pol.slot.FG
    out = {
        "bank": np.zeros(steps, np.int64), "slot": np.zeros(steps, np.int64),
        "zf": np.zeros((steps, Z, FZ), np.float32),
        "af": np.zeros((steps, A, FA), np.float32),
        "gf": np.zeros((steps, FG), np.float32),
        "az": np.zeros((steps, Z, 2), np.float32),
        "aa": np.zeros((steps, A, 3), np.float32),
        "aw": np.zeros((steps, 2), np.float32),
        "rew": np.zeros(steps, np.float32),
        "valid": np.zeros(steps, bool),
        "Z": Z, "A": A, "scenario": W["scenario"],
    }
    for t in range(steps):
        action = E.get_policy_action(pol, obs)
        action = E.validate_action(action, W["spec"])
        rec = pol._record
        if pol._valid and rec is not None:
            out["bank"][t] = rec["bank"]
            out["slot"][t] = rec["slot"]
            out["zf"][t] = rec["zf"]
            out["af"][t] = rec["af"]
            out["gf"][t] = rec["gf"]
            out["az"][t] = rec["az"]
            out["aa"][t] = rec["aa"]
            out["aw"][t] = rec["aw"]
            out["valid"][t] = True
        obs, _, terminated, truncated, _ = tracker.step(action)
        out["rew"][t] = float(tracker.rewards[-1])
        if terminated or truncated:
            for k in ("bank", "slot", "zf", "af", "gf", "az", "aa", "aw",
                      "rew", "valid"):
                out[k] = out[k][:t + 1]
            break
    return out


# ---------------------------------------------------------------- PPO

def compute_gae(rewards, values, gamma, lam):
    """rewards/values [T] (+bootstrap values[-1]); returns adv, ret [T]."""
    T = len(rewards)
    adv = np.zeros(T, np.float64)
    last = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values[:-1]


def compute_mc(rewards, gamma):
    """Monte-Carlo discounted returns (no bootstrap; T >> horizon)."""
    T = len(rewards)
    out = np.zeros(T, np.float64)
    last = 0.0
    for t in reversed(range(T)):
        last = rewards[t] + gamma * last
        out[t] = last
    return out


class SlotPPO:
    """Torch PPO over blended slot heads (+ slot value heads)."""

    def __init__(self, FZ, FA, FG, blend, lr=1e-4, clip=0.2, v_coef=0.5,
                 ent_coef=0.001, min_std=0.02, v_lr=1e-3, v_steps=100):
        import torch
        self.torch = torch
        self.blend = torch.tensor(blend, dtype=torch.float64)
        self.half = len(blend) // 2
        self.clip = clip
        self.v_coef = v_coef
        self.ent_coef = ent_coef
        self.min_logstd = float(np.log(min_std))
        self.v_steps = v_steps
        D = torch.float64
        self.Wz = torch.zeros(2, N_SLOTS, FZ, 2, dtype=D, requires_grad=True)
        self.bz = torch.zeros(2, N_SLOTS, 2, dtype=D, requires_grad=True)
        self.Wa = torch.zeros(2, N_SLOTS, FA, 4, dtype=D, requires_grad=True)
        self.ba = torch.zeros(2, N_SLOTS, 4, dtype=D, requires_grad=True)
        self.Wg = torch.zeros(2, N_SLOTS, FG, 3, dtype=D, requires_grad=True)
        self.bg = torch.zeros(2, N_SLOTS, 3, dtype=D, requires_grad=True)
        self.Wv = torch.zeros(2, N_SLOTS, FG, dtype=D, requires_grad=True)
        self.bv = torch.zeros(2, N_SLOTS, dtype=D, requires_grad=True)
        self.logstd = torch.zeros(3, dtype=D, requires_grad=True)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.vopt = torch.optim.Adam([self.Wv, self.bv], lr=v_lr)

    def parameters(self):
        return [self.Wz, self.bz, self.Wa, self.ba, self.Wg, self.bg,
                self.Wv, self.bv, self.logstd]

    def load_npz(self, path, zero_memory=True):
        z = np.load(path)
        with self.torch.no_grad():
            for k in ("Wz", "bz", "Wa", "ba", "Wg", "bg"):
                getattr(self, k).copy_(
                    self.torch.tensor(z[k], dtype=self.torch.float64))
            if zero_memory:
                self.Wz[:, :, list(ZMEM), :] = 0.0
                self.Wa[:, :, list(AMEM), :] = 0.0
                self.Wg[:, :, list(WMEM), :] = 0.0

    def save_npz(self, path, extra=None):
        arr = {k: getattr(self, k).detach().cpu().numpy()
               for k in ("Wz", "bz", "Wa", "ba", "Wg", "bg", "Wv", "bv")}
        arr["logstd"] = self.logstd.detach().cpu().numpy()
        if extra:
            arr.update(extra)
        np.savez_compressed(path, **arr)

    def _gather(self, W, b, bank, slot):
        """Blend neighboring heads. bank/slot [B] long; W [2,S,F,O]."""
        t = self.torch
        idx = (slot[:, None] + t.arange(
            -self.half, self.half + 1, device=slot.device)) % N_SLOTS
        w = self.blend.to(W.dtype).to(W.device)
        Wm = (W[bank[:, None], idx] * w[None, :, None, None]).sum(1)
        bm = (b[bank[:, None], idx] * w[None, :, None]).sum(1)
        return Wm, bm

    def forward(self, bank, slot, zf, af, gf, zmask, amask):
        """Means for a batch. zf [B,Z,FZ] (+mask [B,Z]), af, gf [B,FG]."""
        t = self.torch
        Wz, bz = self._gather(self.Wz, self.bz, bank, slot)
        Wa, ba = self._gather(self.Wa, self.ba, bank, slot)
        Wg, bg = self._gather(self.Wg, self.bg, bank, slot)
        mz = t.einsum("bzf,bfo->bzo", zf, Wz) + bz[:, None]
        ma = t.einsum("baf,bfo->bao", af, Wa) + ba[:, None]
        mw = t.einsum("bf,bfo->bo", gf, Wg) + bg
        Wv, bv = self._gather(
            self.Wv.unsqueeze(-1), self.bv.unsqueeze(-1), bank, slot)
        val = (gf * Wv.squeeze(-1)).sum(-1) + bv.squeeze(-1)
        return mz, ma, mw, val

    @staticmethod
    def _gauss_logprob(x, mean, logstd):
        return (-0.5 * ((x - mean) / logstd.exp()) ** 2 - logstd
                - 0.5 * np.log(2 * np.pi))

    def value(self, bank, slot, gf):
        t = self.torch
        Wv, bv = self._gather(
            self.Wv.unsqueeze(-1), self.bv.unsqueeze(-1), bank, slot)
        return (gf * Wv.squeeze(-1)).sum(-1) + bv.squeeze(-1)

    def update(self, batch, epochs, minibatch, kl_stop, gamma, gae_lambda):
        t = self.torch
        B = batch["bank"].shape[0]
        # Phase 1: fit the value head on MC returns with its own fast
        # optimizer. A lagging V poisons GAE; the policy lr is far too
        # small to fit cross-scenario return scales (-1..-30).
        for _ in range(self.v_steps):
            vpred = self.value(batch["bank"], batch["slot"], batch["gf"])
            vloss_pre = 0.5 * ((vpred - batch["mc"]) ** 2).mean()
            self.vopt.zero_grad()
            vloss_pre.backward()
            self.vopt.step()
        # Phase 2: GAE with the fitted V, per trajectory bounds.
        with t.no_grad():
            vfull = self.value(
                batch["bank"], batch["slot"], batch["gf"]).cpu().numpy()
        adv = np.zeros(B, np.float64)
        ret = np.zeros(B, np.float64)
        rew = batch["rew"].cpu().numpy()
        for lo, hi in batch["bounds"]:
            v = np.concatenate([vfull[lo:hi], vfull[hi - 1:hi]])
            a, r = compute_gae(rew[lo:hi], v, gamma, gae_lambda)
            adv[lo:hi] = a
            ret[lo:hi] = r
        batch["adv"] = t.tensor(adv, dtype=t.float64)
        batch["ret"] = t.tensor(ret, dtype=t.float64)
        with t.no_grad():
            mz0, ma0, mw0, _ = self.forward(
                batch["bank"], batch["slot"], batch["zf"], batch["af"],
                batch["gf"], batch["zmask"], batch["amask"])
            ls = self.logstd.detach()
            lp0 = (self._gauss_logprob(batch["az"], mz0, ls[0])
                   * batch["zmask"][:, :, None]).sum((1, 2))
            lp0 = lp0 + (self._gauss_logprob(batch["aa"], ma0[..., :3], ls[1])
                         * batch["amask"][:, :, None]).sum((1, 2))
            lp0 = lp0 + self._gauss_logprob(
                batch["aw"], mw0[:, :2], ls[2]).sum(1)
            # Per-dim normalization: ~92 action dims would otherwise blow
            # up summed logprob ratios (clip binds on everything, KL
            # explodes after one minibatch). All PPO mechanics below
            # operate on the mean-field per-dim logprob.
            D = (batch["zmask"].sum(1) * 2 + batch["amask"].sum(1) * 3 + 2)
            lp0 = lp0 / D
        adv = batch["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = batch["ret"]
        idx = t.arange(B)
        total_kl, epochs_run, steps_run = 0.0, 0, 0
        stop = False
        for _ in range(epochs):
            if stop:
                break
            perm = idx[t.randperm(B)]
            kl_acc, nb = 0.0, 0
            for lo in range(0, B, minibatch):
                mb = perm[lo:lo + minibatch]
                mz, ma, mw, val = self.forward(
                    batch["bank"][mb], batch["slot"][mb], batch["zf"][mb],
                    batch["af"][mb], batch["gf"][mb], batch["zmask"][mb],
                    batch["amask"][mb])
                ls = t.clamp(self.logstd, min=self.min_logstd)
                lp = (self._gauss_logprob(batch["az"][mb], mz, ls[0])
                      * batch["zmask"][mb][:, :, None]).sum((1, 2))
                lp = lp + (self._gauss_logprob(batch["aa"][mb], ma[..., :3],
                                               ls[1])
                            * batch["amask"][mb][:, :, None]).sum((1, 2))
                lp = lp + self._gauss_logprob(
                    batch["aw"][mb], mw[:, :2], ls[2]).sum(1)
                lp = lp / D[mb]
                ratio = t.exp(lp - lp0[mb])
                a = adv[mb]
                pg = t.minimum(ratio * a,
                               t.clamp(ratio, 1 - self.clip, 1 + self.clip) * a)
                vloss = 0.5 * ((val - ret[mb]) ** 2).mean()
                ent = ls.mean()  # Gaussian entropy up to a constant
                loss = -pg.mean() + self.v_coef * vloss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                with t.no_grad():
                    self.logstd.clamp_(min=self.min_logstd)
                self.opt.step()
                steps_run += 1
                with t.no_grad():
                    kl_acc += (lp0[mb] - lp).mean().item()
                    nb += 1
                # Per-minibatch trust region: on this sharp landscape a
                # full epoch of steps overshoots before an epoch-end
                # check can stop it.
                if kl_acc / nb > kl_stop:
                    stop = True
                    break
            epochs_run += 1
            total_kl = kl_acc / max(nb, 1)
            if total_kl > kl_stop:
                break
        with t.no_grad():
            vfinal = self.value(batch["bank"], batch["slot"], batch["gf"])
            ev = 1.0 - (((vfinal - batch["ret"]) ** 2).mean()
                        / max(batch["ret"].var().item(), 1e-12))
            ve = float(vfinal.std().item())
        return {"kl": total_kl, "epochs": epochs_run, "steps": steps_run,
                "ent": float(self.logstd.detach().mean().item()),
                "ev": float(max(min(ev, 1.0), -10.0)), "vstd": ve}


def collate(traj, gamma, ppo):
    """Pad + stack worker trajectories; MC returns + bounds (GAE in update)."""
    t = ppo.torch
    maxZ = max(r["Z"] for r in traj)
    maxA = max(r["A"] for r in traj)
    banks, slots, zfs, afs, gfs = [], [], [], [], []
    azs, aas, aws, mcs, rws, zms, ams = [], [], [], [], [], [], []
    bounds = []
    off = 0
    for r in traj:
        mc = compute_mc(r["rew"].astype(np.float64), gamma)
        valid = r["valid"]
        keep = np.where(valid)[0]
        if len(keep) == 0:
            continue
        bounds.append((off, off + len(keep)))
        off += len(keep)
        zpad = np.zeros((len(keep), maxZ, r["zf"].shape[2]), np.float32)
        zpad[:, :r["Z"]] = r["zf"][keep]
        apad = np.zeros((len(keep), maxA, r["af"].shape[2]), np.float32)
        apad[:, :r["A"]] = r["af"][keep]
        azpad = np.zeros((len(keep), maxZ, 2), np.float32)
        azpad[:, :r["Z"]] = r["az"][keep]
        aapad = np.zeros((len(keep), maxA, 3), np.float32)
        aapad[:, :r["A"]] = r["aa"][keep]
        banks.append(r["bank"][keep])
        slots.append(r["slot"][keep])
        zfs.append(zpad)
        afs.append(apad)
        gfs.append(r["gf"][keep])
        azs.append(azpad)
        aas.append(aapad)
        aws.append(r["aw"][keep])
        mcs.append(mc[keep])
        rws.append(r["rew"][keep].astype(np.float64))
        zm = np.zeros((len(keep), maxZ), np.float32)
        zm[:, :r["Z"]] = 1.0
        zms.append(zm)
        am = np.zeros((len(keep), maxA), np.float32)
        am[:, :r["A"]] = 1.0
        ams.append(am)
    cat = lambda xs, dtype: t.tensor(np.concatenate(xs), dtype=dtype)
    return {
        "bank": cat(banks, t.long), "slot": cat(slots, t.long),
        "zf": cat(zfs, t.float64), "af": cat(afs, t.float64),
        "gf": cat(gfs, t.float64), "az": cat(azs, t.float64),
        "aa": cat(aas, t.float64), "aw": cat(aws, t.float64),
        "mc": cat(mcs, t.float64), "rew": cat(rws, t.float64),
        "bounds": bounds,
        "zmask": cat(zms, t.float64), "amask": cat(ams, t.float64),
    }


def resolve_scenarios(name):
    if name == "subset8":
        return SUBSET_8
    if name == "all":
        out = []
        for b in sorted(SCEN_DIR.glob("building_train_*")):
            for w in sorted(b.iterdir()):
                if w.is_dir():
                    out.append((b.name, w.name))
        return out
    return [tuple(p.split("/")) for p in name.split(",")]


def main():
    args = parse_arguments()
    scenarios = resolve_scenarios(args.scenarios)
    print(f"scenarios: {len(scenarios)} workers: {args.workers} "
          f"T={args.rollout_steps} iters={args.iters}", flush=True)

    meta_probe = None
    import json as _json
    b0, w0 = scenarios[0]
    meta_probe = _json.load(open(
        PROJECT_ROOT / "smart_buildings" / "results" / b0 / w0 /
        "scenario_meta.json"))
    from smart_buildings.baselines.slot_linear import SlotFeatures
    probe = SlotFeatures(meta_probe)
    FZ, FA, FG = probe.FZ, probe.FA, probe.FG
    blend = blend_weights((1, 2, 1))
    ppo = SlotPPO(FZ, FA, FG, blend, lr=args.lr, clip=args.clip,
                  v_coef=args.v_coef, ent_coef=args.ent_coef,
                  min_std=args.min_std, v_lr=args.v_lr,
                  v_steps=args.v_steps)
    if args.resume:
        ppo.load_npz(args.resume, zero_memory=False)
        z = np.load(args.resume)
        if "logstd" in z:
            with ppo.torch.no_grad():
                ppo.logstd.copy_(ppo.torch.tensor(
                    z["logstd"], dtype=ppo.torch.float64))
        start_iter = int(_json.load(open(
            RUNS_DIR / f"{args.tag}_meta.json")).get("iter", 0)) + 1 \
            if (RUNS_DIR / f"{args.tag}_meta.json").exists() else 0
    else:
        ppo.load_npz(args.bc, zero_memory=False)
        with ppo.torch.no_grad():
            ppo.logstd.fill_(float(np.log(args.init_std)))
        start_iter = 0
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cur = RUNS_DIR / f"{args.tag}_cur.npz"
    ppo.save_npz(cur, extra={"teacher": "ppo"})

    ctx = mp.get_context("spawn")
    gin_paths = [str(SCEN_DIR / b / w / "config.gin") for b, w in scenarios]
    # One worker per scenario (extra workers cycle scenarios).
    assign = [(scenarios[i % len(scenarios)], gin_paths[i % len(scenarios)])
              for i in range(args.workers)]
    counter = ctx.Value("i", 0)
    pool = ctx.Pool(args.workers, initializer=_worker_init,
                    initargs=(counter, assign))
    logstd = [float(np.log(args.init_std))] * 3
    for it in range(start_iter, args.iters):
        t0 = time.time()
        tasks = [(str(cur), logstd, args.rollout_steps, args.seed + it * 1000 + i)
                 for i in range(args.workers)]
        traj = pool.map(_rollout, tasks, chunksize=1)
        losses = {f"{r['scenario'][0][-1]}/{r['scenario'][1][:4]}":
                  float(-np.mean(r["rew"])) for r in traj}
        mean_loss = float(np.mean(list(losses.values())))
        detail = " ".join(f"{k}={v:.4f}" for k, v in sorted(losses.items()))
        batch = collate(traj, args.gamma, ppo)
        stats = ppo.update(batch, args.epochs, args.minibatch, args.kl_stop,
                           args.gamma, args.gae_lambda)
        logstd = ppo.logstd.detach().cpu().tolist()
        tag = f"{args.tag}_{it:03d}"
        extra = {"teacher": "ppo", "rate_a": np.load(args.bc)["rate_a"],
                 "rate_w": np.load(args.bc)["rate_w"]}
        ppo.save_npz(RUNS_DIR / f"{tag}.npz", extra=extra)
        ppo.save_npz(cur, extra=extra)
        with open(RUNS_DIR / f"{args.tag}_meta.json", "w") as f:
            _json.dump({"iter": it, "mean_loss": mean_loss, **stats}, f)
        el = time.time() - t0
        print(f"iter {it}: loss={mean_loss:.5f} kl={stats['kl']:.4f} "
              f"ep={stats['epochs']} st={stats['steps']} ev={stats['ev']:.3f} "
              f"vs={stats['vstd']:.2f} "
              f"std={np.exp(logstd[0]):.3f} {el:.0f}s", flush=True)
        print(f"   {detail}", flush=True)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()

