"""Transferable bounded coefficient prior; experimental, not a controller.

The model receives a single population coefficient vector, never a building or
scenario identifier. Fold construction and exclusion belong to offline tools.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from smart_buildings.baselines.thermal_greybox import ThermalGreyBoxModel

FEATURE_LOWER = np.array([0, 0, 0, 0, -1, -1, 0, 0], dtype=float)
FEATURE_UPPER = np.array([2, 1, 2, 2, 1, 1, 0.98, 1], dtype=float)
DEFAULT_PRIOR = np.array([0, 0, 0, 0, 0, 0, 0.5, 0], dtype=float)


def fit_population_prior(examples, *, ridge_strength=0.01):
    """Fit eight shared coefficients with equal total weight per saved case.

    Each example is (X, y). No IDs, held-out targets, feature scaling fitted on
    held-out data, or future observations of a test case are accepted here.
    """
    if not np.isfinite(ridge_strength) or ridge_strength <= 0:
        raise ValueError('ridge_strength must be finite and positive')
    gram = ridge_strength * np.eye(8)
    rhs = ridge_strength * DEFAULT_PRIOR.copy()
    count = 0
    for x, y in examples:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or x.shape[1] != 8 or y.shape != (len(x),) or not len(x):
            raise ValueError('Each training case requires nonempty X[n,8] and y[n]')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Training data must be finite')
        gram += x.T @ x / len(x)
        rhs += x.T @ y / len(x)
        count += 1
    if not count:
        raise ValueError('No training cases remain after exclusion')
    # Sufficient statistics keep the actual bounded fit at just 8 x 8.
    # ||L.T theta - b||^2 has the same variable terms as the full weighted fit.
    factor = np.linalg.cholesky(gram)
    fit = lsq_linear(factor.T, np.linalg.solve(factor, rhs),
                     bounds=(FEATURE_LOWER, FEATURE_UPPER))
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f'Population fit failed: {fit.message}')
    return fit.x


class PooledGreyBoxModel(ThermalGreyBoxModel):
    """Population prior, optionally adapted from only the observed test prefix."""

    def __init__(self, metadata, *, population_prior, fit_window=72,
                 adapt=True, local_ridge_strength=16.0):
        super().__init__(metadata, fit_window=fit_window,
                         physical_water=False, thermal_mass=True)
        prior = np.asarray(population_prior, dtype=float)
        if prior.shape != (8,) or not np.isfinite(prior).all():
            raise ValueError('population_prior must be a finite vector of length eight')
        if np.any(prior < FEATURE_LOWER) or np.any(prior > FEATURE_UPPER):
            raise ValueError('population_prior violates physical coefficient bounds')
        if not np.isfinite(local_ridge_strength) or local_ridge_strength <= 0:
            raise ValueError('local_ridge_strength must be finite and positive')
        self.adapt = adapt
        self.ridge_strength = local_ridge_strength
        self.feature_prior = prior.copy()
        self.coefficients = {room: prior.copy() for room in self.rooms}

    def _fit_room(self, room):
        if self.adapt:
            super()._fit_room(room)
