"""Duty-cycled AHUs on the occupancy-gated chain (experimental).

The 30% outside-air ventilation floor applies whenever an AHU runs. In
occupied Fairbanks hours ventilation is over half the gas; in Phoenix hours
it is over half the chiller load. Zone thermal mass smooths short air-side
interruptions, so an AHU whose zones all sit inside their band with margin
can coast with the fan off part of the time: ventilation and fan energy drop
in proportion to the off share, while envelope losses (set by zone
temperature, unchanged) stay the same.

  OccDutyPolicy
    H. Per-AHU duty cycling in occupied steady state: an AHU with every
       served zone inside [lo + MARGIN_K, hi - MARGIN_K] alternates ON_STEPS
       on / OFF_STEPS off. Any zone leaving the inner band restarts the ON
       phase immediately. Catch-up (zones outside the inner band), prestart,
       floor hold and shutdown are untouched.

All signals are live observations and metadata; no scenario identities.
"""

import numpy as np

from smart_buildings.baselines.occ_gated4 import OccGated4Policy


class OccDutyPolicy(OccGated4Policy):
    """OccGated4 + per-AHU duty cycling when occupied and in band."""

    KEY = "occ_duty"

    DUTY_MARGIN_K = 0.7    # inner-band margin each side
    DUTY_ON_STEPS = 4      # 20 min on
    DUTY_OFF_STEPS = 2     # 10 min off

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        self._duty_phase: dict[int, str] = {}   # ahu idx -> "on" | "off"
        self._duty_steps: dict[int, int] = {}   # steps left in this phase

    def _zones_in_margin(self, obs, ctrl) -> bool:
        lo_c, hi_c = self.comfort_temp_window
        for vav_id, obs_idx, _ in ctrl["zones"]:
            lo, hi = self._zone_band(obs, vav_id, lo_c, hi_c)
            temp = self._denormalize_zone_temp(obs[obs_idx])
            if temp < lo + self.DUTY_MARGIN_K or temp > hi - self.DUTY_MARGIN_K:
                return False
        return True

    def _apply_duty(self, obs, action, occupied: bool) -> None:
        if not occupied:
            self._duty_phase = {}
            self._duty_steps = {}
            return
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        for i, ctrl in enumerate(self.ahu_controllers):
            if not ctrl["zone_obs_indices"]:
                continue
            if not self._zones_in_margin(obs, ctrl):
                # Outside the inner band: continuous conditioning, cycle reset.
                self._duty_phase[i] = "on"
                self._duty_steps[i] = self.DUTY_ON_STEPS
                continue
            phase = self._duty_phase.get(i)
            left = self._duty_steps.get(i, 0)
            if phase is None:
                phase, left = "on", self.DUTY_ON_STEPS
            elif left <= 0:
                if phase == "on":
                    phase, left = "off", self.DUTY_OFF_STEPS
                else:
                    phase, left = "on", self.DUTY_ON_STEPS
            self._duty_phase[i] = phase
            self._duty_steps[i] = left - 1
            if phase == "off":
                run = ctrl["run_discrete_idx"]
                disc[run] = 0
                if ctrl["temp_idx"] is not None:
                    cont[ctrl["temp_idx"]] = np.float32(-1.0)
                if ctrl["pressure_idx"] is not None:
                    cont[ctrl["pressure_idx"]] = np.float32(-1.0)
                if ctrl.get("recirc_idx") is not None:
                    cont[ctrl["recirc_idx"]] = np.float32(1.0)
                for _, _, action_indices in ctrl["zones"]:
                    if not action_indices:
                        continue
                    damper_idx = action_indices.get(
                        "supply_air_damper_percentage_command")
                    if damper_idx is not None:
                        cont[damper_idx] = np.float32(-1.0)
                    reheat_idx = action_indices.get(
                        "reheat_valve_percentage_command")
                    if reheat_idx is not None:
                        cont[reheat_idx] = np.float32(-1.0)

    def _chain_action(self, obs):
        action = super()._chain_action(obs)
        # Duty-cycle only in occupied steady state: never during the latched
        # prestart (catch-up needs full power) or the floor hold. The flags
        # are current here (the parent sets them before calling the chain).
        # A stale cycle after shutdown self-heals: drifted zones miss the
        # margin check and restart the ON phase.
        occupied = self._occupied(obs)
        steady = occupied and not self._prestart_active and self._floor_mode is None
        self._apply_duty(obs, action, steady)
        return action


class OccDutyLeanPolicy(OccDutyPolicy):
    """Longer coast: 20 min on / 20 min off."""

    KEY = "occ_duty_lean"
    DUTY_ON_STEPS = 4
    DUTY_OFF_STEPS = 4
