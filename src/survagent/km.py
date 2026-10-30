"""Kaplan–Meier estimation with Greenwood variance and log-log confidence intervals."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

from ._validate import as_survival_arrays, check_probability


def kaplan_meier(
    time: Sequence[float], event: Sequence[int], conf_level: float = 0.95
) -> dict:
    """Kaplan–Meier survival curve.

    Confidence intervals use the log(-log S) transform with Greenwood's variance
    (the default in R's ``survfit(conf.type = "log-log")``), which keeps limits in
    [0, 1] and has better small-sample coverage than the plain interval.

    Returns a dict with the step function (one row per distinct event time) and
    the median survival time with its confidence interval.
    """
    t, e = as_survival_arrays(time, event)
    conf_level = check_probability("conf_level", conf_level)
    z = stats.norm.ppf(0.5 + conf_level / 2)

    event_times = np.unique(t[e == 1])
    n_risk = np.array([(t >= u).sum() for u in event_times], dtype=float)
    n_event = np.array([((t == u) & (e == 1)).sum() for u in event_times], dtype=float)

    surv = np.cumprod(1.0 - n_event / n_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        greenwood_terms = n_event / (n_risk * (n_risk - n_event))
    cum_gw = np.cumsum(greenwood_terms)  # inf once S reaches 0

    lower = np.full_like(surv, np.nan)
    upper = np.full_like(surv, np.nan)
    ok = (surv > 0) & (surv < 1) & np.isfinite(cum_gw)
    log_s = np.log(surv[ok])
    se_loglog = np.sqrt(cum_gw[ok]) / np.abs(log_s)
    lower[ok] = surv[ok] ** np.exp(z * se_loglog)
    upper[ok] = surv[ok] ** np.exp(-z * se_loglog)
    lower[surv == 0] = 0.0
    upper[surv == 0] = 0.0

    median = _first_time_at_or_below(event_times, surv, 0.5, average_ties=True)
    # Median CI by inverting the pointwise bands (as in R's survfit): the lower
    # limit is the first time the *lower* band reaches 0.5, the upper limit the
    # first time the *upper* band does.
    med_lower = _first_time_at_or_below(event_times, np.nan_to_num(lower, nan=1.0), 0.5)
    med_upper = _first_time_at_or_below(event_times, np.nan_to_num(upper, nan=1.0), 0.5)

    notes = []
    if median is None:
        notes.append(
            "Median survival not reached: the estimated survival never falls to 0.5."
        )
    if med_upper is None and median is not None:
        notes.append("Upper confidence limit for the median is not estimable (not reached).")
    if e.sum() < 10:
        notes.append(f"Only {int(e.sum())} events: the curve and its CIs are imprecise.")

    return {
        "method": "Kaplan–Meier; Greenwood variance; log-log CI",
        "conf_level": conf_level,
        "n": int(t.size),
        "events": int(e.sum()),
        "table": [
            {
                "time": float(u),
                "n_risk": int(r),
                "n_event": int(d),
                "survival": float(s),
                "ci_lower": _f(lo),
                "ci_upper": _f(hi),
            }
            for u, r, d, s, lo, hi in zip(event_times, n_risk, n_event, surv, lower, upper)
        ],
        "median": median,
        "median_ci": [med_lower, med_upper],
        "notes": notes,
    }


def survival_at(km_result: dict, t0: float) -> dict:
    """Read the KM estimate and CI at time ``t0`` from a ``kaplan_meier`` result."""
    rows = [r for r in km_result["table"] if r["time"] <= t0]
    if not rows:
        return {"time": t0, "survival": 1.0, "ci_lower": None, "ci_upper": None}
    r = rows[-1]
    return {"time": t0, "survival": r["survival"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"]}


def _first_time_at_or_below(times, curve, level, average_ties=False):
    idx = np.flatnonzero(curve <= level + 1e-12)
    if idx.size == 0:
        return None
    k = idx[0]
    # If the curve sits exactly at `level` over an interval, report its midpoint
    # (R's convention for the median).
    if average_ties and abs(curve[k] - level) < 1e-12 and k + 1 < len(times):
        return float((times[k] + times[k + 1]) / 2)
    return float(times[k])


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)
