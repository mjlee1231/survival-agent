"""Simulation study: do the tools deliver the operating characteristics they claim?

Each check compares a Monte Carlo estimate with its nominal target. A check
passes when the estimate is within 3 Monte Carlo standard errors of the target.

    python validation/run_validation.py            # 2000 replicates per check
    python validation/run_validation.py --reps 500 # quicker

Writes validation/report.md and validation/results.json.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from survagent import __version__
from survagent.cox import cox_regression
from survagent.design import sample_size, simulate_trial, wilson_interval
from survagent.km import kaplan_meier, survival_at
from survagent.logrank import logrank_test

OUT = Path(__file__).resolve().parent
DESIGN = dict(median_control=12.0, accrual=24.0, followup=12.0)  # months


def _check(name, description, hits, reps, target, extra=None):
    est = hits / reps
    mcse = math.sqrt(target * (1 - target) / reps)
    lo, hi = wilson_interval(hits, reps)
    return {
        "check": name,
        "description": description,
        "target": target,
        "estimate": est,
        "mc_ci_95": [lo, hi],
        "mc_se": mcse,
        "passed": abs(est - target) <= 3 * mcse,
        **(extra or {}),
    }


def logrank_type1(rng, reps, n=200):
    hits = 0
    for _ in range(reps):
        t, e, arm = simulate_trial(rng, n, 1.0, **DESIGN)
        hits += logrank_test(t, e, arm)["p_value"] < 0.05
    return _check("logrank_type1_error", f"Log-rank rejection rate under H0 (HR = 1), n = {n}",
                  hits, reps, 0.05)


def cox_coverage(rng, reps, n=200, hr=0.7):
    hits, coefs, ses = 0, [], []
    for _ in range(reps):
        t, e, arm = simulate_trial(rng, n, hr, **DESIGN)
        c = cox_regression(t, e, {"arm": arm})["coefficients"][0]
        hits += c["hr_ci_lower"] <= hr <= c["hr_ci_upper"]
        coefs.append(c["coef"])
        ses.append(c["se"])
    coefs = np.array(coefs)
    extra = {
        "true_log_hr": math.log(hr),
        "mean_log_hr": float(coefs.mean()),
        "bias": float(coefs.mean() - math.log(hr)),
        "empirical_se": float(coefs.std(ddof=1)),
        "mean_model_se": float(np.mean(ses)),
    }
    return _check("cox_hr_ci_coverage", f"Coverage of 95% Wald CI for HR = {hr}, n = {n}",
                  hits, reps, 0.95, extra)


def km_coverage(rng, reps, n=100, t0=12.0):
    true_s = 0.5  # exponential with median 12, evaluated at t0 = 12
    hits = 0
    for _ in range(reps):
        t, e, _ = simulate_trial(rng, n, 1.0, **DESIGN)
        s = survival_at(kaplan_meier(t, e), t0)
        hits += s["ci_lower"] is not None and s["ci_lower"] <= true_s <= s["ci_upper"]
    return _check("km_ci_coverage", f"Coverage of 95% log-log CI for S({t0:g}) = {true_s}, n = {n}",
                  hits, reps, 0.95)


def design_power(rng, reps, hr=0.7):
    ss = sample_size(hr, **DESIGN, power=0.8)
    n = ss["total_n"]
    hits, events = 0, []
    for _ in range(reps):
        t, e, arm = simulate_trial(rng, n, hr, **DESIGN)
        hits += logrank_test(t, e, arm)["p_value"] < 0.05
        events.append(int(e.sum()))
    return _check(
        "designed_power",
        f"Simulated power of the design from sample_size (HR = {hr}, N = {n}); target 0.80",
        hits, reps, 0.80,
        {"total_n": n, "target_events": ss["events"], "mean_simulated_events": float(np.mean(events))},
    )


def _delayed_effect_trial(rng, n, delay=6.0, hr_after=0.5):
    """Treatment has no effect for `delay` months, then HR = hr_after (non-PH)."""
    arm = rng.integers(0, 2, n)
    lam = math.log(2) / DESIGN["median_control"]
    t1 = rng.exponential(1 / lam, n)
    late = (arm == 1) & (t1 > delay)
    t = np.where(late, delay + rng.exponential(1 / (lam * hr_after), n), t1)
    entry = rng.uniform(0, DESIGN["accrual"], n)
    admin = DESIGN["accrual"] + DESIGN["followup"] - entry
    return np.minimum(t, admin), (t <= admin).astype(int), arm


def ph_test(rng, reps, n=400):
    size_hits = power_hits = 0
    for _ in range(reps):
        t, e, arm = simulate_trial(rng, n, 0.7, **DESIGN)
        size_hits += cox_regression(t, e, {"arm": arm})["proportional_hazards_test"]["global"]["p_value"] < 0.05
        t, e, arm = _delayed_effect_trial(rng, n)
        power_hits += cox_regression(t, e, {"arm": arm})["proportional_hazards_test"]["global"]["p_value"] < 0.05
    size = _check("ph_test_size", f"PH-test rejection rate when PH holds (HR = 0.7), n = {n}",
                  size_hits, reps, 0.05)
    lo, hi = wilson_interval(power_hits, reps)
    power = {
        "check": "ph_test_power_delayed_effect",
        "description": f"PH-test rejection rate with a delayed effect (HR 1 for 6 months, then 0.5), n = {n}",
        "target": None,
        "estimate": power_hits / reps,
        "mc_ci_95": [lo, hi],
        "passed": None,
        "note": "Descriptive: how often the PH check flags a realistic non-proportional effect.",
    }
    return [size, power]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261030)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start = time.time()
    results = [
        logrank_type1(rng, args.reps),
        cox_coverage(rng, args.reps),
        km_coverage(rng, args.reps),
        design_power(rng, args.reps),
        *ph_test(rng, args.reps),
    ]
    meta = {
        "package": f"survagent {__version__}",
        "reps": args.reps,
        "seed": args.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - start, 1),
        "design": DESIGN,
    }
    (OUT / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    (OUT / "report.md").write_text(render(meta, results))
    for r in results:
        status = {True: "PASS", False: "FAIL", None: "INFO"}[r["passed"]]
        print(f"[{status}] {r['check']}: {r['estimate']:.3f}"
              + (f" (target {r['target']})" if r["target"] is not None else ""))


def render(meta, results):
    lines = [
        "# Validation report",
        "",
        f"{meta['package']} · {meta['reps']} replicates per check · seed {meta['seed']} · "
        f"run {meta['run_at']} ({meta['seconds']} s)",
        "",
        "Simulated two-arm trials: exponential survival, control median "
        f"{meta['design']['median_control']:g} months, uniform accrual over "
        f"{meta['design']['accrual']:g} months, {meta['design']['followup']:g} months of further "
        "follow-up, 1:1 randomisation. A check passes when the estimate is within 3 Monte Carlo "
        "standard errors of its target.",
        "",
        "| Check | Target | Estimate | 95% MC interval | Result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        status = {True: "Pass", False: "**Fail**", None: "Info"}[r["passed"]]
        target = "—" if r["target"] is None else f"{r['target']:.2f}"
        lines.append(f"| {r['description']} | {target} | {r['estimate']:.3f} | "
                     f"{r['mc_ci_95'][0]:.3f}–{r['mc_ci_95'][1]:.3f} | {status} |")
    cox = next(r for r in results if r["check"] == "cox_hr_ci_coverage")
    dp = next(r for r in results if r["check"] == "designed_power")
    lines += [
        "",
        "## Details",
        "",
        f"- **Cox estimator:** mean log HR {cox['mean_log_hr']:.4f} vs true {cox['true_log_hr']:.4f} "
        f"(bias {cox['bias']:+.4f}); empirical SE {cox['empirical_se']:.4f} vs mean model SE "
        f"{cox['mean_model_se']:.4f}.",
        f"- **Designed trial:** `sample_size` gave N = {dp['total_n']} for {dp['target_events']} target "
        f"events; simulations averaged {dp['mean_simulated_events']:.1f} events.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
