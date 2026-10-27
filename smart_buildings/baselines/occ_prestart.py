# -*- coding: utf-8 -*-
"""Prestart shape (research item 3): low-flow + below-band prestart.

Our 02:00-06:00 prestart is a max-flow sprint (full PI authority on a cold
building: dampers wide open, reheat + water max) and carries 22.5% of total
loss, gas-dominated. This mixin reshapes ONLY the prestart window:

- $PS_DMP_CAP (default 1.0 = off): clamp every VAV damper_01 to this
  ceiling while _prestart_active (catch-up rate control: less -40C air).
- $PS_TGT (default 0.0 = off): shift the PI error by this many K during
  prestart (negative = consider the job done below band centre, e.g. -1.0
  heats to lo instead of centre; final ramp happens at arrival).
- $PS_SAFETY (default empty = class 1.3): prestart timing safety factor.
- $PS_MARGIN (default empty = class 6): extra prestart steps on top.

Defaults reproduce occ_gated7 exactly.
"""

from smart_buildings.baselines.occ_gated7 import OccGated7Policy


class PrestartShapeMixin:
    """Prestart-only damper cap + target shift for OccGated7Policy."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._ps_dmp_cap = float(_os.environ.get("PS_DMP_CAP", "1.0"))
        self._ps_tgt = float(_os.environ.get("PS_TGT", "0.0"))
        safety = _os.environ.get("PS_SAFETY", "")
        if safety:
            self.PRESTART_SAFETY = float(safety)
        margin = _os.environ.get("PS_MARGIN", "")
        if margin:
            self.PRESTART_MARGIN_STEPS = int(margin)

    def _zone_err(self, obs, temp, lo, hi):
        err = super()._zone_err(obs, temp, lo, hi)
        if self._ps_tgt and getattr(self, "_prestart_active", False):
            err = float(err) - self._ps_tgt
        return err

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not getattr(self, "_prestart_active", False):
            return action
        if self._ps_dmp_cap < 1.0:
            cont = action["continuous_action"]
            cap = 2.0 * self._ps_dmp_cap - 1.0
            for vav_id, indices in self.zone_action_indices.items():
                aidx = indices.get("supply_air_damper_percentage_command")
                if aidx is not None and float(cont[aidx]) > cap:
                    cont[aidx] = cap
        return action


class OccPrestart7Policy(PrestartShapeMixin, OccGated7Policy):
    """occ_gated7 with reshaped (low-flow) prestart window."""

    KEY = "occ_ps7"
