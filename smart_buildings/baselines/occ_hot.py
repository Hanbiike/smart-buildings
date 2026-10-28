# -*- coding: utf-8 -*-
"""Hot-climate SAT depth switch, distilled from sb-t-combo-d.ipynb (0.012).

Their DSA_HOT_Sat12Hot: a cooling-mode AHU drops to the 12 C floor only
while its ventilation air is a load (OAT > mean zone temperature);
otherwise it keeps the demand-SAT setting. Rationale: chiller load is flow
x (T_mix - SAT) with ~30% outside air in the mix, so when ventilation air
is a load, colder supply delivers the same zone cooling with ~30% less air;
when OAT is cooler than return air, airflow brings free cooling through the
economizer and colder supply only wastes it.

Note the tension with our mixedness SAT (mixed handlers stay mild at -0.4
to avoid reheat): this override drives -1.0 whenever OAT > inside,
regardless of mixedness. Empirical question, tested here.

Knobs: $HOT_ON (default 1; 0 = occ7-exact), $HOT_SAT (default -1.0),
$HOT_MARGIN_K (default 0.0: switch once OAT > inside - margin).
"""

import numpy as np

from smart_buildings.baselines.occ_gated7 import OccGated7Policy
from smart_buildings.baselines.zone_pi import _outside_kelvin


class OccHotMixin:
    """OAT-vs-return deep-SAT switch for cooling AHUs."""

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        import os as _os
        self._hot_on = int(_os.environ.get("HOT_ON", "1"))
        self._hot_sat = float(_os.environ.get("HOT_SAT", "-1.0"))
        self._hot_margin = float(_os.environ.get("HOT_MARGIN_K", "0.0"))

    def _base_action(self, obs):
        action = super()._base_action(obs)
        if not self._hot_on or not self._pi_active:
            return action
        cont, disc = action["continuous_action"], action["discrete_action"]
        for ctrl in self.ahu_controllers:
            oat_idx = ctrl.get("oat_idx")
            if oat_idx is None or not disc[ctrl["run_discrete_idx"]]:
                continue
            if float(cont[ctrl["temp_idx"]]) > self.COLD_SAT + 1e-6:
                continue
            try:
                inside = float(np.mean([
                    self._denormalize_zone_temp(obs[j])
                    for j in ctrl["zone_obs_indices"]]))
                oat = _outside_kelvin(self, obs[oat_idx])
            except (TypeError, ValueError):
                continue
            if oat > inside - self._hot_margin:
                cont[ctrl["temp_idx"]] = np.float32(self._hot_sat)
        return action


class OccHot7Policy(OccHotMixin, OccGated7Policy):
    """occ_gated7 with hot-climate deep-SAT switch (T_COMBO_D distillation)."""

    KEY = "occ_hot7"
