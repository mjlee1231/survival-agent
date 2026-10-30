"""Log-rank test for two or more groups."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from ._validate import as_groups, as_survival_arrays


def logrank_test(
    time: Sequence[float], event: Sequence[int], group: Sequence
) -> dict:
    """Mantel–Haenszel log-rank test of equal survival across groups.

    The statistic uses the hypergeometric variance with the tie correction
    (n - d) / (n - 1), matching R's ``survdiff(rho = 0)``.
    """
    t, e = as_survival_arrays(time, event)
    g = as_groups(group, t.size)
    levels = sorted(np.unique(g).tolist())
    k = len(levels)
    member = np.stack([g == lv for lv in levels], axis=1).astype(float)  # n x k

    event_times = np.unique(t[e == 1])
    observed = np.zeros(k)
    expected = np.zeros(k)
    var = np.zeros((k, k))
    for u in event_times:
        at_risk = t >= u
        died = (t == u) & (e == 1)
        n_g = member[at_risk].sum(axis=0)
        d_g = member[died].sum(axis=0)
        n, d = n_g.sum(), d_g.sum()
        observed += d_g
        expected += d * n_g / n
        if n > 1:
            frac = n_g / n
            var += d * (n - d) / (n - 1) * (np.diag(frac) - np.outer(frac, frac))

    diff = observed - expected
    stat = float(diff[:-1] @ np.linalg.solve(var[:-1, :-1], diff[:-1]))
    df = k - 1
    p = float(stats.chi2.sf(stat, df))

    notes = []
    small = [lv for lv, ex in zip(levels, expected) if ex < 5]
    if small:
        notes.append(
            "Expected events below 5 in group(s) "
            + ", ".join(small)
            + ": the chi-square approximation may be poor."
        )
    if e.sum() == 0:
        raise ValueError("There are no events, so the log-rank test is undefined.")

    return {
        "method": "Log-rank (Mantel–Haenszel) test",
        "statistic": stat,
        "df": df,
        "p_value": p,
        "groups": [
            {"group": lv, "n": int(member[:, i].sum()), "observed": float(observed[i]), "expected": float(expected[i])}
            for i, lv in enumerate(levels)
        ],
        "notes": notes,
    }
