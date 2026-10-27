"""Grey-box per-zone thermal model (identification + horizon prediction).

The model is a lumped RC balance per VAV zone, identified online from the
observations the policy itself sees:

    T[t+1] - T[t] = a*(T_out[t] - T[t])
                   + b*flow[t]*(T_sup[t] - T[t])
                   + c*flow[t]*reheat[t]*(T_water[t] - T[t])
                   + d*occ[t]
                   + e

* ``T_out``    outside air temperature (raw kelvin, handler sensor)
* ``T_sup``    handler supply air temperature (raw kelvin, handler sensor)
* ``flow``     zone airflow fraction = damper command x handler running state
* ``reheat``   zone reheat valve opening (0..1, VAV sensor)
* ``T_water``  hot water supply temperature (raw kelvin, HWS sensor)
* ``occ``      building occupancy

Only observable quantities feed the model, so the same code runs offline on
the banked observation dumps and online inside a policy.

Two estimator modes:

* ``rls``       - recursive least squares with a forgetting factor, warm
                  started from a fixed prior (online, no warm-up window).
* ``batch_rls`` - ridge least squares over the first ``fit_window`` steps,
                  then RLS from that solution (online, one-day warm-up).

The model is a *tool*, not a controller: it predicts what a zone will do and
how long a warm-up needs, and the control layer decides what to do about it.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

import numpy as np

__all__ = ["GreyBoxModel", "GreyBoxParams"]

# Regressor layout of the zone balance.
P_ENV = 0      # a: envelope coupling to outside air
P_SUP = 1      # b: supply-air coupling
P_RHT = 2      # c: reheat (water) coupling
P_OCC = 3      # d: internal gains from occupancy
P_CONST = 4    # e: constant term
N_PARAMS = 5


def _trailing_room_number(device_token: str) -> int | None:
    match = re.search(r"(\d+)\s*$", str(device_token))
    return int(match.group(1)) if match else None


class GreyBoxParams:
    """Per-zone identified parameters and trust state."""

    __slots__ = ("theta", "pmat", "residual", "bias", "n_obs", "ready")

    def __init__(self, theta: np.ndarray, pmat: np.ndarray) -> None:
        self.theta = theta
        self.pmat = pmat
        self.residual = 0.0     # EMA of |1-step error| (kelvin)
        self.bias = 0.0          # signed EMA of (actual - predicted)
        self.n_obs = 0
        self.ready = False

    def copy_theta(self) -> np.ndarray:
        return self.theta.copy()


class GreyBoxModel:
    """Online grey-box identification of every zone's thermal response."""

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        *,
        mode: str = "batch_rls",
        rls_forgetting: float = 0.999,
        fit_window: int = 144,
        ridge: float = 1e-3,
        prior: tuple[float, ...] = (0.004, 0.02, 0.02, 0.0, 0.0),
        prior_weight: float = 25.0,
        residual_ema: float = 0.05,
        zone_temp_mean: float = 295.0,
        zone_temp_std: float = 5.0,
    ) -> None:
        self.metadata = dict(building_meta_data)
        self.mode = mode
        self.rls_forgetting = float(rls_forgetting)
        self.fit_window = int(fit_window)
        self.ridge = float(ridge)
        self.prior = np.asarray(prior, dtype=np.float64)
        self.prior_weight = float(prior_weight)
        self.residual_ema = float(residual_ema)

        names = list(self.metadata['observation_field_names'])
        self.field_pos = {n: i for i, n in enumerate(names)}
        norm = dict(self.metadata.get('observation_normalization_metadata') or {})

        # --- zone observations -------------------------------------------
        self.room_temp_obs: dict[int, int] = {}
        self.room_damper_obs: dict[int, int] = {}
        self.room_reheat_obs: dict[int, int] = {}
        for name, pos in self.field_pos.items():
            if name.endswith('_zone_air_temperature_sensor'):
                room = _trailing_room_number(name[:-len('_zone_air_temperature_sensor')])
                if room is not None:
                    self.room_temp_obs[room] = pos
            elif name.endswith('_supply_air_damper_percentage_command'):
                room = _trailing_room_number(name[:-len('_supply_air_damper_percentage_command')])
                if room is not None:
                    self.room_damper_obs[room] = pos
            elif name.endswith('_reheat_valve_percentage_command'):
                room = _trailing_room_number(name[:-len('_reheat_valve_percentage_command')])
                if room is not None:
                    self.room_reheat_obs[room] = pos

        temp_meta = norm.get('zone_air_temperature_sensor') or {}
        self.zone_temp_mean = float(temp_meta.get('sample_mean', zone_temp_mean) or zone_temp_mean)
        self.zone_temp_std = math.sqrt(
            max(float(temp_meta.get('sample_variance', 25.0) or 25.0), 1e-9)
        ) if not temp_meta else float(temp_meta.get('sample_std', 5.0) or 5.0)

        damper_meta = norm.get('supply_air_damper_percentage_command') or {}
        self.damper_mean = float(damper_meta.get('sample_mean', 0.5) or 0.5)
        self.damper_std = float(damper_meta.get('sample_std', 0.5) or 0.5)

        reheat_meta = norm.get('reheat_valve_percentage_command') or {}
        self.reheat_mean = float(reheat_meta.get('sample_mean', 0.5) or 0.5)
        self.reheat_std = float(reheat_meta.get('sample_std', 0.5) or 0.5)

        # --- plant observations ------------------------------------------
        self.ambient_idx = self._first_suffix('_outside_air_temperature_sensor')
        self.occupancy_idx = self.field_pos.get('num_occupants')
        self.water_temp_idx = self._first_suffix('_supply_water_temperature_sensor')

        self.ahu_supply_obs: dict[int, int] = {}
        self.ahu_run_obs: dict[int, int] = {}
        for name, pos in self.field_pos.items():
            match = re.search(r'ahu_(\d+)_supply_air_temperature_sensor$', name)
            if match:
                self.ahu_supply_obs.setdefault(int(match.group(1)), pos)
            match = re.search(r'ahu_(\d+)_supervisor_run_command$', name)
            if match:
                self.ahu_run_obs.setdefault(int(match.group(1)), pos)

        # --- room -> handler ----------------------------------------------
        assignments = dict(self.metadata.get('ahu_zone_assignments') or {})
        self.room_ahu: dict[int, int] = {}
        self.ahu_ids: list[int] = []
        for ahu_id, rooms in assignments.items():
            num = _trailing_room_number(ahu_id)
            if num is None:
                continue
            self.ahu_ids.append(num)
            for room in rooms:
                rn = _trailing_room_number(str(room))
                if rn is not None:
                    self.room_ahu[rn] = num
        self.ahu_ids.sort()

        # --- model state ---------------------------------------------------
        self.params: dict[int, GreyBoxParams] = {}
        self._batch_xtx: dict[int, np.ndarray] = {}
        self._batch_xty: dict[int, np.ndarray] = {}
        self._prev: dict[int, dict[str, float]] | None = None
        self._prev_ambient = 0.0
        self._prev_occ = 0.0
        self.step_count = 0

        for room in self.room_temp_obs:
            theta = self.prior.copy()
            pmat = np.eye(N_PARAMS, dtype=np.float64) * self.prior_weight
            self.params[room] = GreyBoxParams(theta, pmat)
            self._batch_xtx[room] = np.zeros((N_PARAMS, N_PARAMS), dtype=np.float64)
            self._batch_xty[room] = np.zeros(N_PARAMS, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _first_suffix(self, suffix: str) -> int | None:
        for name, pos in self.field_pos.items():
            if name.endswith(suffix):
                return pos
        return None

    def zone_temp(self, obs: np.ndarray, room: int) -> float:
        idx = self.room_temp_obs.get(room)
        if idx is None:
            return self.zone_temp_mean
        return float(obs[idx]) * self.zone_temp_std + self.zone_temp_mean

    def _damper_native(self, obs: np.ndarray, room: int) -> float:
        idx = self.room_damper_obs.get(room)
        if idx is None:
            return 0.0
        return float(np.clip(float(obs[idx]) * self.damper_std + self.damper_mean, 0.0, 1.0))

    def _reheat_native(self, obs: np.ndarray, room: int) -> float:
        idx = self.room_reheat_obs.get(room)
        if idx is None:
            return 0.0
        return float(np.clip(float(obs[idx]) * self.reheat_std + self.reheat_mean, 0.0, 1.0))

    def _ahu_running(self, obs: np.ndarray, ahu: int) -> float:
        idx = self.ahu_run_obs.get(ahu)
        if idx is None:
            return 1.0
        return 1.0 if float(obs[idx]) > 0.5 else 0.0

    def _ahu_supply(self, obs: np.ndarray, ahu: int) -> float:
        idx = self.ahu_supply_obs.get(ahu)
        if idx is None:
            return self.zone_temp_mean
        return float(obs[idx])

    def _ambient(self, obs: np.ndarray) -> float:
        if self.ambient_idx is None:
            return self.zone_temp_mean
        return float(obs[self.ambient_idx])

    def _water_temp(self, obs: np.ndarray) -> float:
        if self.water_temp_idx is None:
            return self.zone_temp_mean
        return float(obs[self.water_temp_idx])

    def _occupancy(self, obs: np.ndarray) -> float:
        if self.occupancy_idx is None:
            return 0.0
        return float(obs[self.occupancy_idx])

    # ------------------------------------------------------------------ #
    # identification
    # ------------------------------------------------------------------ #

    def _regressors(self, snap: Mapping[str, float]) -> np.ndarray:
        t_zone = snap['temp']
        x = np.empty(N_PARAMS, dtype=np.float64)
        x[P_ENV] = snap['ambient'] - t_zone
        x[P_SUP] = snap['flow'] * (snap['t_sup'] - t_zone)
        x[P_RHT] = snap['flow'] * snap['reheat'] * (snap['t_water'] - t_zone)
        x[P_OCC] = snap['occ']
        x[P_CONST] = 1.0
        return x

    def _snapshot(self, obs: np.ndarray) -> dict[int, dict[str, float]]:
        ambient = self._ambient(obs)
        occ = self._occupancy(obs)
        water = self._water_temp(obs)
        out: dict[int, dict[str, float]] = {}
        for room, idx in self.room_temp_obs.items():
            ahu = self.room_ahu.get(room)
            flow = 0.0
            t_sup = self.zone_temp_mean
            if ahu is not None:
                run = self._ahu_running(obs, ahu)
                flow = self._damper_native(obs, room) * run
                t_sup = self._ahu_supply(obs, ahu)
            out[room] = {
                'temp': float(obs[idx]) * self.zone_temp_std + self.zone_temp_mean,
                'flow': flow,
                't_sup': t_sup,
                't_water': water,
                'reheat': self._reheat_native(obs, room),
                'ambient': ambient,
                'occ': occ,
            }
        return out

    def update(self, obs: np.ndarray) -> None:
        """Absorb one observation: identify the transition from the previous one."""
        snap_now = self._snapshot(obs)
        if self._prev is not None:
            for room, prev in self._prev.items():
                now = snap_now.get(room)
                if now is None:
                    continue
                params = self.params[room]
                x = self._regressors(prev)
                y = now['temp'] - prev['temp']
                params.n_obs += 1
                if self.mode == 'batch_rls' and not params.ready:
                    self._batch_xtx[room] += np.outer(x, x)
                    self._batch_xty[room] += x * y
                    if params.n_obs >= self.fit_window:
                        ridge = self.ridge * max(1.0, float(np.trace(self._batch_xtx[room])))
                        try:
                            theta = np.linalg.solve(
                                self._batch_xtx[room] + ridge * np.eye(N_PARAMS),
                                self._batch_xty[room],
                            )
                        except np.linalg.LinAlgError:
                            theta = params.theta
                        params.theta = theta
                        params.pmat = np.linalg.pinv(
                            self._batch_xtx[room] + ridge * np.eye(N_PARAMS)
                        ) * self.prior_weight
                        params.ready = True
                    continue
                # RLS step with forgetting.
                px = params.pmat @ x
                denom = self.rls_forgetting + float(x @ px)
                k_gain = px / max(denom, 1e-9)
                y_hat = float(x @ params.theta)
                error = y - y_hat
                params.theta = params.theta + k_gain * error
                params.pmat = (
                    params.pmat - np.outer(k_gain, px)
                ) / self.rls_forgetting
                params.residual = (
                    (1.0 - self.residual_ema) * params.residual
                    + self.residual_ema * abs(error)
                )
                params.bias = (
                    (1.0 - self.residual_ema) * params.bias
                    + self.residual_ema * error
                )
        self._prev = snap_now
        self.step_count += 1

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        room: int,
        steps: int,
        *,
        temp0: float | None = None,
        hold: Mapping[str, float] | None = None,
    ) -> float:
        """Predict the zone temperature ``steps`` steps ahead (held inputs)."""
        params = self.params.get(room)
        if params is None or self._prev is None:
            return temp0 if temp0 is not None else self.zone_temp_mean
        snap = self._prev[room]
        t = float(temp0 if temp0 is not None else snap['temp'])
        if hold:
            ambient = float(hold.get('ambient', snap['ambient']))
            t_sup = float(hold.get('t_sup', snap['t_sup']))
            t_water = float(hold.get('t_water', snap['t_water']))
            flow = float(hold.get('flow', snap['flow']))
            reheat = float(hold.get('reheat', snap['reheat']))
            occ = float(hold.get('occ', snap['occ']))
        else:
            ambient = snap['ambient']
            t_sup = snap['t_sup']
            t_water = snap['t_water']
            flow = snap['flow']
            reheat = snap['reheat']
            occ = snap['occ']
        a, b, c, d, e = [float(v) for v in params.theta]
        for _ in range(max(0, int(steps))):
            t = t + (
                a * (ambient - t)
                + b * flow * (t_sup - t)
                + c * flow * reheat * (t_water - t)
                + d * occ
                + e
            )
        return float(t)

    def steps_to_reach(
        self,
        room: int,
        target: float,
        max_steps: int,
        *,
        temp0: float | None = None,
        hold: Mapping[str, float] | None = None,
    ) -> int:
        """Steps of held actuation needed to reach ``target`` (model-simulated).

        Returns ``max_steps`` when the target is not reached in time.
        """
        params = self.params.get(room)
        if params is None or self._prev is None:
            return int(max_steps)
        snap = self._prev[room]
        t = float(temp0 if temp0 is not None else snap['temp'])
        if hold:
            ambient = float(hold.get('ambient', snap['ambient']))
            t_sup = float(hold.get('t_sup', snap['t_sup']))
            t_water = float(hold.get('t_water', snap['t_water']))
            flow = float(hold.get('flow', snap['flow']))
            reheat = float(hold.get('reheat', snap['reheat']))
            occ = float(hold.get('occ', snap['occ']))
        else:
            ambient = snap['ambient']
            t_sup = snap['t_sup']
            t_water = snap['t_water']
            flow = snap['flow']
            reheat = snap['reheat']
            occ = snap['occ']
        a, b, c, d, e = [float(v) for v in params.theta]
        for step in range(1, int(max_steps) + 1):
            t = t + (
                a * (ambient - t)
                + b * flow * (t_sup - t)
                + c * flow * reheat * (t_water - t)
                + d * occ
                + e
            )
            if t >= target:
                return step
        return int(max_steps)

    def trusted(self, room: int, residual_limit: float = 0.5) -> bool:
        """Conservative trust gate: identified and with a small 1-step error."""
        params = self.params.get(room)
        if params is None or params.n_obs < 24:
            return False
        a, b, c, d, e = [float(v) for v in params.theta]
        plausible = (
            -0.05 <= a <= 0.2
            and -0.5 <= b <= 2.0
            and -0.5 <= c <= 2.0
            and -1.0 <= d <= 1.0
            and abs(e) <= 1.0
        )
        if not plausible:
            return False
        return params.residual < residual_limit

    def summary(self, room: int) -> dict[str, float]:
        params = self.params.get(room)
        if params is None:
            return {}
        a, b, c, d, e = [float(v) for v in params.theta]
        return {
            'env': a, 'sup': b, 'rht': c, 'occ': d, 'const': e,
            'residual': params.residual, 'bias': params.bias,
            'n_obs': float(params.n_obs),
        }
