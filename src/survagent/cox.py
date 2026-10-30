"""Cox proportional hazards regression (Efron or Breslow ties) with a PH check."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from ._validate import as_survival_arrays, check_probability


# ----------------------------------------------------------------------------
# Design matrix
# ----------------------------------------------------------------------------

def build_design(
    covariates: Mapping[str, Sequence], n: int, reference_levels: Mapping[str, str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Turn ``{name: values}`` into a numeric matrix.

    Numeric columns are used as-is. Text columns are treated as categorical and
    dummy-coded against a reference level (the one given in ``reference_levels``,
    otherwise the alphabetically first level).
    """
    if not covariates:
        raise ValueError("Provide at least one covariate.")
    reference_levels = dict(reference_levels or {})
    cols, names = [], []
    for name, values in covariates.items():
        vals = list(values)
        if len(vals) != n:
            raise ValueError(f"Covariate `{name}` has {len(vals)} values but `time` has {n}.")
        if _is_numeric(vals):
            x = np.asarray(vals, dtype=float)
            if not np.all(np.isfinite(x)):
                raise ValueError(f"Covariate `{name}` contains missing or infinite values.")
            if np.ptp(x) == 0:
                raise ValueError(f"Covariate `{name}` is constant, so its effect is not estimable.")
            cols.append(x)
            names.append(name)
        else:
            s = np.asarray([str(v) for v in vals], dtype=object)
            levels = sorted(np.unique(s).tolist())
            if len(levels) < 2:
                raise ValueError(f"Covariate `{name}` has a single level, so its effect is not estimable.")
            ref = reference_levels.get(name, levels[0])
            if ref not in levels:
                raise ValueError(f"Reference level `{ref}` is not a level of `{name}` ({levels}).")
            for lv in levels:
                if lv != ref:
                    cols.append((s == lv).astype(float))
                    names.append(f"{name}={lv} (vs {ref})")
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X - X.mean(axis=0)) < X.shape[1]:
        raise ValueError("Covariates are collinear; remove redundant columns.")
    return X, names


def _is_numeric(vals) -> bool:
    if any(isinstance(v, str) for v in vals):
        return False
    try:
        np.asarray(vals, dtype=float)
        return True
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------
# Partial likelihood
# ----------------------------------------------------------------------------

class _Data:
    """Sorted data plus the index bookkeeping needed for risk-set sums."""

    def __init__(self, t, e, X):
        order = np.argsort(t, kind="stable")
        self.t, self.e, self.X = t[order], e[order], X[order]
        self.event_times = np.unique(self.t[self.e == 1])
        self.first_at_risk = np.searchsorted(self.t, self.event_times, side="left")
        ev = self.e == 1
        self.ev_idx = np.flatnonzero(ev)
        self.ev_time_id = np.searchsorted(self.event_times, self.t[ev])
        self.d = np.bincount(self.ev_time_id, minlength=self.event_times.size)
        J = self.event_times.size
        self.rep = np.repeat(np.arange(J), self.d)
        starts = np.repeat(np.cumsum(self.d) - self.d, self.d)
        self.r = np.arange(self.rep.size) - starts


def _terms(beta, D: _Data, ties: str):
    X = D.X
    p = X.shape[1]
    eta = X @ beta
    shift = eta.max()
    w = np.exp(eta - shift)

    rc0 = np.cumsum(w[::-1])[::-1]
    wx = w[:, None] * X
    rc1 = np.cumsum(wx[::-1], axis=0)[::-1]
    wxx = wx[:, :, None] * X[:, None, :]
    rc2 = np.cumsum(wxx[::-1], axis=0)[::-1]
    S0, S1, S2 = rc0[D.first_at_risk], rc1[D.first_at_risk], rc2[D.first_at_risk]

    J = D.event_times.size
    jid = D.ev_time_id
    T0 = np.bincount(jid, weights=w[D.ev_idx], minlength=J)
    T1 = np.zeros((J, p))
    np.add.at(T1, jid, wx[D.ev_idx])
    T2 = np.zeros((J, p, p))
    np.add.at(T2, jid, wxx[D.ev_idx])

    c = D.r / D.d[D.rep] if ties == "efron" else np.zeros(D.rep.size)
    den = S0[D.rep] - c * T0[D.rep]
    a = (S1[D.rep] - c[:, None] * T1[D.rep]) / den[:, None]
    B = (S2[D.rep] - c[:, None, None] * T2[D.rep]) / den[:, None, None]

    loglik = (eta[D.ev_idx] - shift).sum() - np.log(den).sum()
    score = X[D.ev_idx].sum(axis=0) - a.sum(axis=0)
    info = (B - a[:, :, None] * a[:, None, :]).sum(axis=0)

    abar = np.zeros((J, p))
    np.add.at(abar, D.rep, a)
    abar /= D.d[:, None]
    schoenfeld = X[D.ev_idx] - abar[jid]
    return loglik, score, info, schoenfeld


def _fit(D: _Data, ties: str, max_iter: int = 50, tol: float = 1e-9):
    p = D.X.shape[1]
    beta = np.zeros(p)
    ll, score, info, _ = _terms(beta, D, ties)
    ll0, score0, info0 = ll, score, info
    converged = False
    for it in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError:
            break
        new = beta + step
        ll_new, s_new, i_new, _ = _terms(new, D, ties)
        halvings = 0
        while (not np.isfinite(ll_new) or ll_new < ll - 1e-12) and halvings < 30:
            step /= 2
            new = beta + step
            ll_new, s_new, i_new, _ = _terms(new, D, ties)
            halvings += 1
        beta, ll_prev = new, ll
        ll, score, info = ll_new, s_new, i_new
        if abs(ll - ll_prev) < tol * (abs(ll) + 1):
            converged = True
            break
    return beta, ll, info, ll0, score0, info0, converged, it


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def cox_regression(
    time: Sequence[float],
    event: Sequence[int],
    covariates: Mapping[str, Sequence],
    reference_levels: Mapping[str, str] | None = None,
    ties: str = "efron",
    conf_level: float = 0.95,
) -> dict:
    """Fit a Cox proportional hazards model.

    Reports hazard ratios with Wald confidence intervals; the likelihood-ratio,
    Wald and score tests; and a Grambsch–Therneau test of proportional hazards
    based on scaled Schoenfeld residuals (KM time transform, as in R's
    ``cox.zph`` before survival 3.0).
    """
    t, e = as_survival_arrays(time, event)
    conf_level = check_probability("conf_level", conf_level)
    if ties not in ("efron", "breslow"):
        raise ValueError("`ties` must be 'efron' or 'breslow'.")
    if e.sum() == 0:
        raise ValueError("There are no events, so the Cox model cannot be fitted.")
    X, names = build_design(covariates, t.size, reference_levels)
    D = _Data(t, e, X)
    beta, ll, info, ll0, score0, info0, converged, iters = _fit(D, ties)

    p = len(names)
    V = np.linalg.inv(info)
    se = np.sqrt(np.diag(V))
    z = stats.norm.ppf(0.5 + conf_level / 2)
    zstat = beta / se
    pvals = 2 * stats.norm.sf(np.abs(zstat))

    lr = 2 * (ll - ll0)
    wald = float(beta @ info @ beta)
    score_stat = float(score0 @ np.linalg.solve(info0, score0))

    ph = _ph_test(beta, D, ties, V, names)

    notes = []
    n_events = int(e.sum())
    epv = n_events / p
    if epv < 10:
        notes.append(
            f"{n_events} events for {p} coefficient(s) ({epv:.1f} events per coefficient): "
            "estimates may be unstable or biased; consider fewer covariates."
        )
    if not converged:
        notes.append("Newton–Raphson did not converge; results are unreliable.")
    if np.any(np.abs(beta) > 10):
        notes.append(
            "A coefficient is very large, which suggests monotone likelihood "
            "(e.g. no events in one group); the hazard ratio is not estimable."
        )
    flagged = [r["term"] for r in ph["per_term"] if r["p_value"] < 0.05]
    if flagged:
        notes.append(
            "Proportional hazards may be violated for: " + ", ".join(flagged)
            + ". A single hazard ratio may misrepresent a time-varying effect; inspect "
            "Schoenfeld residuals, or consider stratification or time-varying coefficients."
        )

    return {
        "method": f"Cox proportional hazards ({ties} ties); Wald CIs",
        "conf_level": conf_level,
        "n": int(t.size),
        "events": n_events,
        "coefficients": [
            {
                "term": nm,
                "coef": float(b),
                "se": float(s),
                "hazard_ratio": float(np.exp(b)),
                "hr_ci_lower": float(np.exp(b - z * s)),
                "hr_ci_upper": float(np.exp(b + z * s)),
                "z": float(zs),
                "p_value": float(pv),
            }
            for nm, b, s, zs, pv in zip(names, beta, se, zstat, pvals)
        ],
        "global_tests": {
            "likelihood_ratio": {"statistic": float(lr), "df": p, "p_value": float(stats.chi2.sf(lr, p))},
            "wald": {"statistic": wald, "df": p, "p_value": float(stats.chi2.sf(wald, p))},
            "score": {"statistic": score_stat, "df": p, "p_value": float(stats.chi2.sf(score_stat, p))},
        },
        "proportional_hazards_test": ph,
        "converged": converged,
        "iterations": iters,
        "log_likelihood": float(ll),
        "notes": notes,
    }


def _ph_test(beta, D: _Data, ties, V, names):
    _, _, _, resid = _terms(beta, D, ties)  # unscaled Schoenfeld residuals, one row per event
    d = resid.shape[0]
    # KM transform of event times: g(t) = 1 - S_KM(t)
    n_risk = np.array([(D.t >= u).sum() for u in D.event_times], dtype=float)
    surv = np.cumprod(1 - D.d / n_risk)
    g = 1 - surv[D.ev_time_id]
    gc = g - g.mean()
    denom = (gc ** 2).sum()
    if d < 3 or denom == 0:
        return {"method": "not computed (too few events)", "per_term": [], "global": None}
    u = gc @ resid  # p-vector: covariance of residuals with g(t)
    per = (u @ V) ** 2 * d / (np.diag(V) * denom)  # uses scaled residuals d * V r
    glob = float(u @ V @ u * d / denom)
    p = len(names)
    return {
        "method": "Grambsch–Therneau test, scaled Schoenfeld residuals vs 1 - KM(t)",
        "per_term": [
            {"term": nm, "chisq": float(c), "df": 1, "p_value": float(stats.chi2.sf(c, 1))}
            for nm, c in zip(names, per)
        ],
        "global": {"chisq": glob, "df": p, "p_value": float(stats.chi2.sf(glob, p))},
    }
