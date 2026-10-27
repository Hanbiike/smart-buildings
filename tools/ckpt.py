"""Checkpoint / restore for branching search over the building simulator.

Spike-validated: strip the lazy SuperLU factorization cache, then
``copy.deepcopy((env, tracker, policy))`` in ONE call (shared references such
as tracker->env are preserved by the memo). Restoring and continuing for 100
steps is bit-identical to the uninterrupted run (rewards exact, final obs
exact). Snapshot cost: ~1 s, ~91 MB pickled (building_train_1).

The factorization is a pure lazy cache of (static matrices, convection
coefficient): clearing it only forces one deterministic rebuild on the next
step (see FiniteDifferenceSolver.solve).
"""

from __future__ import annotations

import copy
from typing import Any


def strip_caches(root: Any) -> int:
    """Clear lazy SuperLU caches anywhere under ``root``. Returns count."""
    seen: set[int] = set()
    cleared = 0
    stack = [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, dict):
            stack.extend(o.values())
            continue
        if isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
            continue
        d = getattr(o, "__dict__", None)
        if d is None:
            continue
        if "_factorization" in d:
            d["_factorization"] = None
            d["_factorized_at"] = None
            cleared += 1
        stack.extend(d.values())
    return cleared


def snapshot(env: Any, tracker: Any, policy: Any) -> tuple[Any, Any, Any]:
    """Take a restorable snapshot of the full sim state."""
    strip_caches((env, tracker, policy))
    return copy.deepcopy((env, tracker, policy))


def restore(master: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
    """Fresh independent copy of a snapshot (one per branch candidate)."""
    return copy.deepcopy(master)


def reseed_convection(stream_seed: int) -> None:
    """Reseed the convection kernel RNG (common random numbers).

    The numba shuffle kernel draws from a process-global RNG that no
    deepcopy can capture, so consecutive branch rollouts would otherwise
    see different shuffle realizations. Reseeding every branch to the same
    stream makes candidates exactly comparable; the stream choice itself
    moves rewards only ~1e-12 relative (shuffles preserve room means).
    Call once per branch, right after restore(), before stepping.
    """
    from smart_buildings.smart_control.simulator import (
        stochastic_convection_simulator as scs,
    )

    scs._seed_kernel_rng(int(stream_seed) % (2**31 - 1))
