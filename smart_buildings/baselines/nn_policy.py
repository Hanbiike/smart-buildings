# -*- coding: utf-8 -*-
"""NN policy wrapping the behavior-cloned windowed controller (bc_v0).

Imitation-trained on gen-3.1 (`dsat_gen31`) trajectories; swaps in for the
rule stack when the validation gate improves on it. Requirements match the
competition: wiring derives only from `building_meta_data` + the observation
layout, actions are produced per step with a single tiny CPU forward pass
(d=64 transformer over window tokens; torch CPU, no GPU needed).

Runtime flow per `get_action(obs)`:

  1. decode features via the shared `FeatureContext` (same code as training);
  2. push them into a 72-step ring buffer, pool 12 history slots at stride 6;
  3. one ControllerNet forward (batch B=1);
  4. heads map back through the same room/AHU index wiring the policies use:
     per-room damper/reheat, per-AHU SAT/pressure/recirc + run bit, global
     water temp/pressure + pump bit.

Before the window is full (first 72 steps) the pooled slots clamp to step 0,
exactly like the training tokenizer, so warmup is consistent with the
teacher's true first steps.
"""
from pathlib import Path

import numpy as np
import torch

from smart_buildings.nn_model import ControllerNet, FeatureContext

CHECKPOINT = (Path(__file__).resolve().parents[2]
              / "smart_buildings" / "results" / "nn_runs" / "bc_v0.pt")


class BcV0Policy:
    """Behavior-cloned competition policy; `policy = BcV0Policy(meta, spec)`."""

    CHECKPOINT = CHECKPOINT
    # Optional post-processing of raw regression outputs (class attrs; off by
    # default). SAT_MODES snaps the AHU supply-temperature command to the
    # teacher's branch values (the dump histogram is strongly multimodal:
    # deep-cool/mid/neutral/max-heat branches), turning a smeared regression
    # back into a branch choice. MIN_RUN_HOLD latches run bits to avoid fan
    # chatter below the teacher's natural switching rate.
    SAT_MODES: tuple | None = None
    MIN_RUN_HOLD: int = 0

    def __init__(self, building_meta_data, action_spec, checkpoint=None):
        self.ctx = FeatureContext(building_meta_data)
        meta = building_meta_data
        self.field_names = meta["observation_field_names"]
        self.n_discrete = int(meta["n_discrete_action"])
        self.n_continuous = int(meta["n_continuous_action"])

        ckpt_path = checkpoint or self.CHECKPOINT
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        cfg = ckpt["args"]
        self.model = ControllerNet(ckpt["fz"], ckpt["fa"], ckpt["fg"],
                                   d=cfg["d_model"], layers=cfg["layers"],
                                   heads=cfg["heads"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # ring buffer of pooled history inputs (room-mean + ahu-mean + global)
        fh = self.model.proj_hist.in_features
        self.hist_buf = np.zeros((0, fh), dtype=np.float32)

        # index wiring
        self.damp_act = [r["damp_act"] for r in self.ctx.rooms]
        self.reheat_act = [r["reheat_act"] for r in self.ctx.rooms]
        self.ahu_act = self.ctx.ahu_actions
        self.water = self.ctx.water
        # run-bit latch state (used when MIN_RUN_HOLD > 0)
        self.run_bits = [None] * len(self.ahu_act)
        self.run_hold = [0] * len(self.ahu_act)

    def reset(self) -> None:
        self.hist_buf = np.zeros((0, self.hist_buf.shape[1]), dtype=np.float32)
        self.run_bits = [None] * len(self.ahu_act)
        self.run_hold = [0] * len(self.ahu_act)

    def get_action(self, obs):
        ctx = self.ctx
        zf_np = ctx.room_feats(obs)                 # [Z, Fz]
        af_np = ctx.ahu_feats(obs)                  # [A, Fa]
        gf_np = ctx.glob_feats(obs)                 # [Fg]
        return self._infer(zf_np, af_np, gf_np)

    def _infer(self, zf_np, af_np, gf_np):
        """Ring-buffer pooling + one forward + heads -> action dict.

        Inputs are already-shaped feature tables ([Z, Fz], [A, Fa], [Fg]) so
        subclasses can fold extra channels (e.g. teacher-shadow state) into
        them before inference.
        """
        row = np.concatenate([zf_np.mean(axis=0), af_np.mean(axis=0), gf_np])
        self.hist_buf = np.append(self.hist_buf, row[None], axis=0)[-80:]
        H = 12  # N_HIST
        S = 6
        n = len(self.hist_buf)
        hist_rows = np.stack([
            self.hist_buf[max(0, n - (H - k) * S)] for k in range(H)
        ])

        zf = torch.from_numpy(zf_np).unsqueeze(0)   # [1, Z, Fz]
        af = torch.from_numpy(af_np).unsqueeze(0)   # [1, A, Fa]
        gf = torch.from_numpy(gf_np).unsqueeze(0)   # [1, Fg]
        hp = torch.from_numpy(hist_rows).unsqueeze(0)  # [1, H, Fh]

        with torch.no_grad():
            out = self.model(zf, af, gf, hp)
        return self._compose(out)

    def _compose(self, out: dict) -> dict:
        """Heads -> env action dict via the shared index wiring."""
        cont = np.zeros(self.n_continuous, dtype=np.float32)
        cont[:] = -1.0                       # neutral default everywhere
        disc = np.zeros(self.n_discrete, dtype=np.int64)

        zone = out["zone"][0].numpy()        # [Z, 2]
        for k, (di, ri) in enumerate(zip(self.damp_act, self.reheat_act)):
            cont[di] = float(np.clip(zone[k, 0], -1.0, 1.0))
            cont[ri] = float(np.clip(zone[k, 1], -1.0, 1.0))

        ahu = out["ahu_sat"].numpy()[0]
        press = out["ahu_pressure"].numpy()[0]
        recirc = out["ahu_recirc"].numpy()[0]
        run_logits = out["ahu_run_logit"].numpy()[0]
        for i, act in enumerate(self.ahu_act):
            sat = float(np.clip(ahu[i], -1.0, 1.0))
            if self.SAT_MODES:
                modes = np.asarray(self.SAT_MODES)
                sat = float(modes[int(np.argmin(np.abs(modes - sat)))])
            cont[act["temp_idx"]] = sat
            if act["pressure_idx"] is not None:
                cont[act["pressure_idx"]] = float(np.clip(press[i], -1.0, 1.0))
            if act["recirc_idx"] is not None:
                cont[act["recirc_idx"]] = float(np.clip(recirc[i], -1.0, 1.0))
            bit = 1 if run_logits[i] > 0 else 0
            if (self.MIN_RUN_HOLD > 0
                    and self.run_hold[i] > 0
                    and self.run_hold[i] < self.MIN_RUN_HOLD
                    and self.run_bits[i] is not None):
                bit = self.run_bits[i]
            if bit != self.run_bits[i]:
                self.run_bits[i] = bit
                self.run_hold[i] = 0
            self.run_hold[i] += 1
            disc[act["run_disc_idx"]] = bit

        w = self.water
        if w["water_temp_idx"] is not None:
            cont[w["water_temp_idx"]] = float(
                np.clip(out["water"][0].item(), -1.0, 1.0))
        if w["water_pressure_idx"] is not None:
            cont[w["water_pressure_idx"]] = float(
                np.clip(out["water_pressure"][0].item(), -1.0, 1.0))
        if 0 <= w["water_discrete_idx"] < self.n_discrete:
            disc[w["water_discrete_idx"]] = (
                1 if out["water_run_logit"][0].item() > 0 else 0)

        return {"continuous_action": cont, "discrete_action": disc}


class DaggerPolicy:
    """Execute the NN policy; record the teacher's action for the same obs.

    Companion to `dump_policy_trajectory`: the results dir receives
    obs_dump.npz (obs + executed NN action) plus dagger_labels.npz (the
    teacher's action for each executed obs), feeding DAgger round 2 —
    imitation data on the NN's own state distribution instead of the
    teacher's.
    """

    NN_CLASS = BcV0Policy

    def __init__(self, building_meta_data, action_spec):
        from smart_buildings.baselines.zone_pi import DsatGen31Policy

        self.nn = self.NN_CLASS(building_meta_data, action_spec)
        self.teacher = DsatGen31Policy(building_meta_data, action_spec)
        self.teacher_labels: list[np.ndarray] = []

    def reset(self) -> None:
        self.nn.reset()
        self.teacher_labels = []

    def _dump_state(self):
        return self.teacher._dump_state()

    def get_action(self, obs):
        action = self.nn.get_action(obs)
        teacher_action = self.teacher.get_action(obs)
        self.teacher_labels.append(np.concatenate([
            np.asarray(teacher_action["continuous_action"], dtype=np.float32),
            np.asarray(teacher_action["discrete_action"], dtype=np.float32),
        ]))
        return action


class BcV1Policy(BcV0Policy):
    """DAgger round 2 checkpoint (teacher labels on NN-visited states)."""

    CHECKPOINT = (Path(__file__).resolve().parents[2]
                  / "smart_buildings" / "results" / "nn_runs" / "bc_v1.pt")


class BcV1SnapPolicy(BcV1Policy):
    """BcV1Policy with SAT commands snapped to the teacher's branch modes."""

    SAT_MODES = (-1.0, -0.4, -0.05, 0.95)


class DaggerV2Policy(DaggerPolicy):
    """DAgger round 2 collection: bc_v1 executes, teacher labels + state."""

    NN_CLASS = BcV1Policy


class BcV2Policy(BcV0Policy):
    """v2 controller: exact teacher state channels via a shadow gen-3.1.

    A shadow `DsatGen31Policy` is advanced on every observation (its action
    is ignored) and its bookkeeping — per-zone PI signal/integral, per-AHU
    mixedness/cooling EMAs and hold counters, global PI-hours/occupant EMA —
    is folded into the model features. All channels are deterministic
    functions of our own observation history, so training and deployment
    share the identical code path (`_dump_state`).
    """

    CHECKPOINT = (Path(__file__).resolve().parents[2]
                  / "smart_buildings" / "results" / "nn_runs" / "bc_v2.pt")

    def __init__(self, building_meta_data, action_spec, checkpoint=None):
        from smart_buildings.baselines.zone_pi import DsatGen31Policy

        super().__init__(building_meta_data, action_spec, checkpoint)
        self.shadow = DsatGen31Policy(building_meta_data, action_spec)

    def reset(self) -> None:
        super().reset()
        self.shadow.reset()

    def get_action(self, obs):
        # Advance the shadow teacher (action ignored) and fold its state into
        # the features — same layout as nn_tokenizer's state concat.
        self.shadow.get_action(obs)
        st = self.shadow._dump_state().astype(np.float32)
        ctx = self.ctx
        nz, na = ctx.Z * 2, ctx.A * 3
        z_ex = st[:nz].reshape(ctx.Z, 2).copy()
        z_ex[:, 1] /= 100.0                               # same as tokenizer
        zf_np = np.concatenate([ctx.room_feats(obs), z_ex], axis=1)
        af_np = np.concatenate(
            [ctx.ahu_feats(obs), st[nz:nz + na].reshape(ctx.A, 3)], axis=1)
        gf_np = np.concatenate([ctx.glob_feats(obs), st[nz + na:]])
        return self._infer(zf_np, af_np, gf_np)


class BcV2SnapPolicy(BcV2Policy):
    """BcV2Policy with SAT commands snapped to the teacher's branch modes."""

    SAT_MODES = (-1.0, -0.4, -0.05, 0.95)


class BcV2BPolicy(BcV2Policy):
    """v2 at d=96: more headroom for big buildings (85 zones / 9 AHUs)."""

    CHECKPOINT = (Path(__file__).resolve().parents[2]
                  / "smart_buildings" / "results" / "nn_runs" / "bc_v2b.pt")


class BcV2EnsemblePolicy(BcV2Policy):
    """Average the heads of two v2 checkpoints (d=64 + d=96).

    The two models underfit different buildings; averaging continuous heads
    cancels their distinct closed-loop excursions, and run bits take the
    shared-sign logit mean. Checkpoints stay separate; the ensemble only
    changes inference.
    """

    SECOND_CHECKPOINT = (Path(__file__).resolve().parents[2]
                         / "smart_buildings" / "results" / "nn_runs" / "bc_v2b.pt")

    def __init__(self, building_meta_data, action_spec, checkpoint=None):
        super().__init__(building_meta_data, action_spec, checkpoint)
        ckpt_path = checkpoint or self.SECOND_CHECKPOINT
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        cfg = ckpt["args"]
        self.model_b = ControllerNet(ckpt["fz"], ckpt["fa"], ckpt["fg"],
                                     d=cfg["d_model"], layers=cfg["layers"],
                                     heads=cfg["heads"])
        self.model_b.load_state_dict(ckpt["model"])
        self.model_b.eval()
        if self.model_b.proj_hist.in_features != self.model.proj_hist.in_features:
            raise ValueError("ensemble checkpoints must share the input dims")

    def _infer(self, zf_np, af_np, gf_np):
        row = np.concatenate([zf_np.mean(axis=0), af_np.mean(axis=0), gf_np])
        self.hist_buf = np.append(self.hist_buf, row[None], axis=0)[-80:]
        H, S, n = 12, 6, len(self.hist_buf)
        hist_rows = np.stack([
            self.hist_buf[max(0, n - (H - k) * S)] for k in range(H)
        ])
        zf = torch.from_numpy(zf_np).unsqueeze(0)
        af = torch.from_numpy(af_np).unsqueeze(0)
        gf = torch.from_numpy(gf_np).unsqueeze(0)
        hp = torch.from_numpy(hist_rows).unsqueeze(0)
        with torch.no_grad():
            out_a = self.model(zf, af, gf, hp)
            out_b = self.model_b(zf, af, gf, hp)
        out = {k: 0.5 * (out_a[k] + out_b[k]) for k in out_a}
        return self._compose(out)


class DaggerV3Policy(DaggerPolicy):
    """DAgger round 3 collection: bc_v2 ensemble executes, teacher labels."""

    NN_CLASS = BcV2EnsemblePolicy


class BcV3EnsemblePolicy(BcV2Policy):
    """Mean-of-heads ensemble over N v2-line checkpoints.

    Generalizes BcV2EnsemblePolicy: CHECKPOINTS lists every member; all
    continuous heads average, run bits take the mean logit sign. Each
    checkpoint (torch.load, weights_only=True) must share input dims or the
    init raises.
    """

    CHECKPOINTS = ()

    def __init__(self, building_meta_data, action_spec, checkpoint=None):
        super().__init__(building_meta_data, action_spec, checkpoint)
        self.models = [self.model]
        for path in self.CHECKPOINTS:
            if str(path) == str(checkpoint or self.CHECKPOINT):
                continue                       # already loaded as self.model
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
            cfg = ckpt["args"]
            m = ControllerNet(ckpt["fz"], ckpt["fa"], ckpt["fg"],
                              d=cfg["d_model"], layers=cfg["layers"],
                              heads=cfg["heads"])
            m.load_state_dict(ckpt["model"])
            m.eval()
            if m.proj_hist.in_features != self.model.proj_hist.in_features:
                raise ValueError("ensemble members must share the input dims")
            self.models.append(m)

    def _infer(self, zf_np, af_np, gf_np):
        row = np.concatenate([zf_np.mean(axis=0), af_np.mean(axis=0), gf_np])
        self.hist_buf = np.append(self.hist_buf, row[None], axis=0)[-80:]
        H, S, n = 12, 6, len(self.hist_buf)
        hist_rows = np.stack([
            self.hist_buf[max(0, n - (H - k) * S)] for k in range(H)
        ])
        zf = torch.from_numpy(zf_np).unsqueeze(0)
        af = torch.from_numpy(af_np).unsqueeze(0)
        gf = torch.from_numpy(gf_np).unsqueeze(0)
        hp = torch.from_numpy(hist_rows).unsqueeze(0)
        outs = []
        with torch.no_grad():
            for m in self.models:
                outs.append(m(zf, af, gf, hp))
        out = {k: sum(o[k] for o in outs) / len(outs) for k in outs[0]}
        return self._compose(out)


def _nn_runs(name: str) -> Path:
    return (Path(__file__).resolve().parents[2] / "smart_buildings"
            / "results" / "nn_runs" / name)


class BcV3Policy(BcV2Policy):
    """Round-3 DAgger member at d=64 (teacher + dagger_v2 + dagger_v3 data)."""

    CHECKPOINT = _nn_runs("bc_v3.pt")


class BcV3BPolicy(BcV2Policy):
    """Round-3 DAgger member at d=96."""

    CHECKPOINT = _nn_runs("bc_v3b.pt")


class BcV4EnsemblePolicy(BcV3EnsemblePolicy):
    """4-way mean of heads: bc_v2 (d64) + bc_v2b (d96) + bc_v3 + bc_v3b."""

    CHECKPOINT = _nn_runs("bc_v3.pt")
    CHECKPOINTS = (_nn_runs("bc_v2.pt"), _nn_runs("bc_v2b.pt"),
                   _nn_runs("bc_v3b.pt"))
