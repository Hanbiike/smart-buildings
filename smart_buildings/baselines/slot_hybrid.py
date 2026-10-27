# -*- coding: utf-8 -*-
"""Hybrid slot controller: rule-based modes/discrete + slot-linear continuous.

Pure linear BC of the stateful teacher fails (run bits collapse, no integral
action, compounding drift into the quadratic comfort tails: 0.06-0.36 vs
0.005-0.05). The discrete structure -- shutdown, latched prestart, floor
hold, AHU/water run bits -- is exactly what BC mangles, so it stays
rule-based (OccGated4). The slot-linear heads learn only the smooth
continuous modulation (dampers, reheat, SAT, pressure, recirculation, water
temperatures/pressures), where least-squares regression is well-posed.

This is also the RL policy parameterization: PPO later finetunes the same
heads from this strong behavior-cloned init.
"""

import numpy as np

from smart_buildings.baselines.occ_gated4 import OccGated4Policy
from smart_buildings.baselines.slot_linear import N_SLOTS, SlotFeatures


class SlotHybridPolicy(OccGated4Policy):
    """OccGated4 modes + run bits; slot heads drive continuous actions."""

    KEY = "slot_hybrid"
    BLEND = (1, 2, 1)
    DEFAULT_WEIGHTS = "smart_buildings/results/slot_runs/slot_bc_occ4.npz"

    def __init__(self, building_meta_data, action_spec, weights_path=None):
        super().__init__(building_meta_data, action_spec)
        from smart_buildings.baselines.slot_linear import blend_weights
        self.slot = SlotFeatures(building_meta_data)
        self.blend = blend_weights(self.BLEND)
        self.half = len(self.blend) // 2
        if weights_path is None:
            weights_path = self.DEFAULT_WEIGHTS
            import os as _os
            if not _os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"slot weights missing: {weights_path} "
                    "(run tools/slot_train.py first)")
        z = np.load(weights_path)
        self.Wz = z["Wz"].astype(np.float64)
        self.bz = z["bz"].astype(np.float64)
        self.Wa = z["Wa"].astype(np.float64)
        self.ba = z["ba"].astype(np.float64)
        self.Wg = z["Wg"].astype(np.float64)
        self.bg = z["bg"].astype(np.float64)

    def reset(self) -> None:
        super().reset()
        # The PID base calls reset() inside __init__, before self.slot
        # exists; guard so construction still works.
        slot = getattr(self, "slot", None)
        if slot is not None:
            slot.reset()

    def _blended(self, W, b, bank, slot):
        Wm = np.zeros_like(W[bank, slot], dtype=np.float64)
        bm = np.zeros_like(b[bank, slot], dtype=np.float64)
        for j, w in enumerate(self.blend):
            s = (slot + j - self.half) % N_SLOTS
            Wm += w * W[bank, s]
            bm += w * b[bank, s]
        return Wm, bm

    def _chain_action(self, obs):
        action = super()._chain_action(obs)
        bank, slot, (zf, af, gf) = self.slot.step(np.asarray(obs))
        Wz, bz = self._blended(self.Wz, self.bz, bank, slot)
        Wa, ba = self._blended(self.Wa, self.ba, bank, slot)
        Wg, bg = self._blended(self.Wg, self.bg, bank, slot)
        zone = zf.astype(np.float64) @ Wz + bz
        ahu = af.astype(np.float64) @ Wa + ba
        wat = gf.astype(np.float64) @ Wg + bg
        cont = action["continuous_action"]
        rooms = self.slot.ctx.rooms
        for k, r in enumerate(rooms):
            cont[r["damp_act"]] = np.float32(np.clip(zone[k, 0], -1.0, 1.0))
            cont[r["reheat_act"]] = np.float32(np.clip(zone[k, 1], -1.0, 1.0))
        for i, act in enumerate(self.slot.ctx.ahu_actions):
            cont[act["temp_idx"]] = np.float32(np.clip(ahu[i, 0], -1.0, 1.0))
            if act["pressure_idx"] is not None:
                cont[act["pressure_idx"]] = np.float32(
                    np.clip(ahu[i, 1], -1.0, 1.0))
            if act["recirc_idx"] is not None:
                cont[act["recirc_idx"]] = np.float32(
                    np.clip(ahu[i, 2], -1.0, 1.0))
        w = self.slot.ctx.water
        if w["water_temp_idx"] is not None:
            cont[w["water_temp_idx"]] = np.float32(np.clip(wat[0], -1.0, 1.0))
        if w["water_pressure_idx"] is not None:
            cont[w["water_pressure_idx"]] = np.float32(
                np.clip(wat[1], -1.0, 1.0))
        return action


class SlotPPOPolicy(SlotHybridPolicy):
    """Deployment head for PPO checkpoints (deterministic, no explorer).

    Weights path comes from $SLOT_PPO_WEIGHTS so full-episode evals and the
    submission notebook can point at any checkpoint without code changes.
    The npz layout matches SlotHybridPolicy (extra Wv/bv/logstd ignored).
    """

    KEY = "slot_ppo"

    def __init__(self, building_meta_data, action_spec, weights_path=None):
        if weights_path is None:
            import os as _os
            weights_path = _os.environ.get(
                "SLOT_PPO_WEIGHTS",
                "smart_buildings/results/slot_runs/slot_ppo_cur.npz")
        super().__init__(building_meta_data, action_spec, weights_path)
