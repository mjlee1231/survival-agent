"""Sample size and power for two-arm trials analysed with the log-rank test.

Analytic results use Schoenfeld's (1981) approximation for the number of events,
and an exponential model with uniform accrual for the probability of an event.
``simulate_power`` checks the analytic answer by Monte Carlo.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ._validate import check_positive, check_probability
from .logrank import logrank_test


def events_required(
    hazard_ratio: float, alpha: float = 0.05, power: float = 0.8, allocation: float = 0.5
) -> dict:
    """Number of events for a two-sided log-rank test (Schoenfeld formula).

    ``allocation`` is the fraction of patients randomised to the experimental arm.
    """
    hr = check_positive("hazard_ratio", hazard_ratio)
    if hr == 1:
        raise ValueError("`hazard_ratio` = 1 means no effect; the required events are infinite.")
    alpha = check_probability("alpha", alpha)
    power = check_probability("power", power)
    p = check_probability("allocation", allocation)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    d = (za + zb) ** 2 / (p * (1 - p) * math.log(hr) ** 2)
    return {
        "method": "Schoenfeld (1981) approximation, two-sided log-rank test",
        "events": int(math.ceil(d)),
        "events_exact": d,
        "inputs": {"hazard_ratio": hr, "alpha": alpha, "power": power, "allocation": p},
        "notes": _hr_notes(hr),
    }


def power_from_events(
    hazard_ratio: float, events: int, alpha: float = 0.05, allocation: float = 0.5
) -> dict:
    """Approximate power of the two-sided log-rank test given the number of events."""
    hr = check_positive("hazard_ratio", hazard_ratio)
    alpha = check_probability("alpha", alpha)
    p = check_probability("allocation", allocation)
    d = check_positive("events", events)
    za = stats.norm.ppf(1 - alpha / 2)
    shift = math.sqrt(d * p * (1 - p)) * abs(math.log(hr))
    pw = stats.norm.cdf(shift - za) + stats.norm.cdf(-shift - za)
    return {
        "method": "Schoenfeld (1981) approximation, two-sided log-rank test",
        "power": float(pw),
        "inputs": {"hazard_ratio": hr, "events": d, "alpha": alpha, "allocation": p},
        "notes": _hr_notes(hr),
    }


def event_probability(hazard: float, accrual: float, followup: float) -> float:
    """P(event observed) under exponential survival, uniform accrual over ``accrual``
    and additional follow-up ``followup`` after accrual ends (no dropout)."""
    lam = hazard
    if accrual == 0:
        return 1 - math.exp(-lam * followup)
    return 1 - (math.exp(-lam * followup) - math.exp(-lam * (accrual + followup))) / (lam * accrual)


def sample_size(
    hazard_ratio: float,
    median_control: float,
    accrual: float,
    followup: float,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation: float = 0.5,
) -> dict:
    """Total patients needed so that the expected number of events reaches the
    Schoenfeld target. Time units (e.g. months) must match across arguments."""
    ev = events_required(hazard_ratio, alpha, power, allocation)
    med = check_positive("median_control", median_control)
    acc = float(accrual)
    fu = check_positive("followup", followup)
    if acc < 0:
        raise ValueError("`accrual` must be non-negative.")
    lam_c = math.log(2) / med
    lam_t = lam_c * ev["inputs"]["hazard_ratio"]
    p = ev["inputs"]["allocation"]
    pe_c = event_probability(lam_c, acc, fu)
    pe_t = event_probability(lam_t, acc, fu)
    pe = (1 - p) * pe_c + p * pe_t
    n = int(math.ceil(ev["events_exact"] / pe))
    notes = list(ev["notes"]) + [
        "Assumes exponential survival, uniform accrual and no loss to follow-up; "
        "inflate the sample size for expected dropout.",
        "Check this design by simulation (simulate_power) before relying on it.",
    ]
    return {
        "method": "Schoenfeld events / P(event) under exponential survival with uniform accrual",
        "total_n": n,
        "n_per_arm": {"control": int(math.ceil(n * (1 - p))), "experimental": int(math.ceil(n * p))},
        "events": ev["events"],
        "event_probability": {"control": pe_c, "experimental": pe_t, "overall": pe},
        "inputs": {**ev["inputs"], "median_control": med, "accrual": acc, "followup": fu},
        "notes": notes,
    }


def simulate_trial(
    rng: np.random.Generator,
    n: int,
    hazard_ratio: float,
    median_control: float,
    accrual: float,
    followup: float,
    allocation: float = 0.5,
):
    """One simulated two-arm trial: exponential survival, uniform accrual,
    administrative censoring at the end of the study."""
    arm = (rng.random(n) < allocation).astype(int)
    lam = math.log(2) / median_control * np.where(arm == 1, hazard_ratio, 1.0)
    t_true = rng.exponential(1 / lam)
    entry = rng.uniform(0, accrual, n) if accrual > 0 else np.zeros(n)
    admin = accrual + followup - entry
    time = np.minimum(t_true, admin)
    event = (t_true <= admin).astype(int)
    return time, event, arm


def simulate_power(
    hazard_ratio: float,
    n_total: int,
    median_control: float,
    accrual: float,
    followup: float,
    alpha: float = 0.05,
    allocation: float = 0.5,
    n_sim: int = 1000,
    seed: int | None = 2026,
) -> dict:
    """Monte Carlo power of the log-rank test, with a 95% Wilson interval for the
    Monte Carlo error."""
    hr = check_positive("hazard_ratio", hazard_ratio)
    n = int(check_positive("n_total", n_total))
    alpha = check_probability("alpha", alpha)
    p = check_probability("allocation", allocation)
    n_sim = int(n_sim)
    if not 100 <= n_sim <= 20000:
        raise ValueError("`n_sim` must be between 100 and 20000.")
    rng = np.random.default_rng(seed)
    rejections, events = 0, []
    for _ in range(n_sim):
        time, event, arm = simulate_trial(rng, n, hr, median_control, accrual, followup, p)
        if event.sum() == 0 or arm.min() == arm.max():
            continue
        res = logrank_test(time, event, arm)
        rejections += res["p_value"] < alpha
        events.append(int(event.sum()))
    lo, hi = wilson_interval(rejections, n_sim)
    return {
        "method": "Monte Carlo simulation of the two-sided log-rank test",
        "power": rejections / n_sim,
        "mc_ci_95": [lo, hi],
        "n_sim": n_sim,
        "mean_events": float(np.mean(events)) if events else 0.0,
        "seed": seed,
        "inputs": {"hazard_ratio": hr, "n_total": n, "median_control": median_control,
                   "accrual": accrual, "followup": followup, "alpha": alpha, "allocation": p},
        "notes": ["Assumes exponential survival, uniform accrual and no loss to follow-up."],
    }


def wilson_interval(successes: int, n: int, conf_level: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + conf_level / 2)
    phat = successes / n
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _hr_notes(hr: float) -> list[str]:
    if 0.95 < hr < 1.05:
        return ["The hazard ratio is very close to 1, so the required study size is very large."]
    return []
