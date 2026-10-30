"""Input validation shared by every statistical function.

Every public function validates its inputs here and raises ``ValueError`` with a
message written for a human (or an AI agent) to act on.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def as_survival_arrays(
    time: Sequence[float], event: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert follow-up times and event indicators."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event)
    if t.ndim != 1 or e.ndim != 1:
        raise ValueError("`time` and `event` must be one-dimensional lists.")
    if t.size == 0:
        raise ValueError("`time` is empty.")
    if t.size != e.size:
        raise ValueError(
            f"`time` has {t.size} values but `event` has {e.size}; they must match."
        )
    if not np.all(np.isfinite(t)):
        raise ValueError("`time` contains missing or infinite values.")
    if np.any(t < 0):
        raise ValueError("`time` must be non-negative.")
    try:
        e = e.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("`event` must contain only 0 (censored) and 1 (event).") from exc
    if not np.all(np.isin(e, (0.0, 1.0))):
        raise ValueError("`event` must contain only 0 (censored) and 1 (event).")
    return t, e.astype(int)


def as_groups(group: Sequence, n: int) -> np.ndarray:
    g = np.asarray([str(x) for x in group], dtype=object)
    if g.size != n:
        raise ValueError(f"`group` has {g.size} values but `time` has {n}; they must match.")
    if np.unique(g).size < 2:
        raise ValueError("`group` must contain at least two distinct groups.")
    return g


def check_probability(name: str, value: float, *, open_interval: bool = True) -> float:
    v = float(value)
    ok = 0 < v < 1 if open_interval else 0 <= v <= 1
    if not ok:
        raise ValueError(f"`{name}` must be strictly between 0 and 1 (got {value}).")
    return v


def check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"`{name}` must be a positive number (got {value}).")
    return v
