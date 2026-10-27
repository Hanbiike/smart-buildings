"""First baseline: average-zone control, sensor comfort bands, decoded weather.

Builds on the Apache-2.0 organizer PIDWaterResetPolicy (see vendor LICENSE).
Uses only metadata and live observations; no scenario identities or weather files.
"""
import numpy as np
from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy


class BaselinePolicy(PIDWaterResetPolicy):
    USE_SENSOR_BAND = True
    # Worst-zone aggregation increased loss in both full-length pilot scenarios.
    USE_WORST_ZONE = False

    @property
    def name(self):
        return "transfer"

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        stats = self.observation_normalization_metadata.get(
            "outside_air_temperature_sensor", {}
        )
        self.outside_mean = float(stats.get("sample_mean", 0.0))
        self.outside_std = float(stats.get("sample_std", 1.0))

    def _recirculation(self, obs, ctrl, u, low, high):
        # The parent's economizer expects kelvin; the observation is standardized.
        idx = ctrl.get("oat_idx")
        if idx is None:
            return 1.0
        decoded = obs.copy()
        decoded[idx] = obs[idx] * self.outside_std + self.outside_mean
        return super()._recirculation(decoded, ctrl, u, low, high)

    def get_action(self, obs):
        action = super().get_action(obs)
        # Enforce the gateway's dtype, shape and bounds contract.
        for key, dtype in (("continuous_action", np.float32),
                           ("discrete_action", np.int32)):
            space = self.action_spec[key]
            action[key] = np.clip(action[key], space.low, space.high).astype(dtype)
        return action
