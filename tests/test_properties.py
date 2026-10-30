"""Mathematical properties and input validation."""

import numpy as np
import pytest

from survagent.cox import cox_regression
from survagent.design import events_required, power_from_events, sample_size
from survagent.km import kaplan_meier
from survagent.logrank import logrank_test


def _sim(rng, n=200, hr=0.7):
    arm = rng.integers(0, 2, n)
    t = rng.exponential(1 / np.where(arm == 1, hr, 1.0))
    c = rng.exponential(2.0, n)
    return np.minimum(t, c), (t <= c).astype(int), arm


def test_km_without_censoring_is_empirical_survival():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    km = kaplan_meier(t, np.ones(4, int))
    assert [r["survival"] for r in km["table"]] == pytest.approx([0.75, 0.5, 0.25, 0.0])


def test_km_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    t, e, _ = _sim(rng)
    for r in kaplan_meier(t, e)["table"]:
        if r["ci_lower"] is not None:
            assert r["ci_lower"] <= r["survival"] <= r["ci_upper"]


def test_cox_score_test_close_to_logrank_without_ties():
    rng = np.random.default_rng(2)
    t, e, arm = _sim(rng)
    lr = logrank_test(t, e, arm)["statistic"]
    score = cox_regression(t, e, {"arm": arm})["global_tests"]["score"]["statistic"]
    assert score == pytest.approx(lr, rel=1e-6)


def test_cox_is_invariant_to_row_order():
    rng = np.random.default_rng(3)
    t, e, arm = _sim(rng)
    perm = rng.permutation(t.size)
    a = cox_regression(t, e, {"arm": arm})["coefficients"][0]["coef"]
    b = cox_regression(t[perm], e[perm], {"arm": arm[perm]})["coefficients"][0]["coef"]
    assert a == pytest.approx(b, abs=1e-10)


def test_cox_reference_level_flips_sign(leukemia):
    cov = {"treatment": list(leukemia.treatment)}
    a = cox_regression(leukemia.time, leukemia.event, cov)["coefficients"][0]["coef"]
    b = cox_regression(leukemia.time, leukemia.event, cov,
                       reference_levels={"treatment": "placebo"})["coefficients"][0]["coef"]
    assert a == pytest.approx(-b, abs=1e-8)


def test_cox_warns_on_few_events_per_covariate():
    rng = np.random.default_rng(4)
    t, e, arm = _sim(rng, n=30)
    covs = {f"x{i}": rng.normal(size=30) for i in range(4)}
    notes = cox_regression(t, e, covs)["notes"]
    assert any("events per coefficient" in n for n in notes)


def test_schoenfeld_events_textbook_value():
    # HR 0.7, two-sided alpha 0.05, 80% power, 1:1 -> 247 events
    assert events_required(0.7)["events"] == 247


def test_power_inverts_events_required():
    d = events_required(0.75, power=0.9)["events_exact"]
    assert power_from_events(0.75, d)["power"] == pytest.approx(0.9, abs=1e-3)


def test_sample_size_exceeds_events():
    res = sample_size(0.7, median_control=12, accrual=24, followup=12)
    assert res["total_n"] > res["events"]
    assert 0 < res["event_probability"]["overall"] < 1


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"time": [1, 2], "event": [1]}, "must match"),
        ({"time": [1, -2], "event": [1, 0]}, "non-negative"),
        ({"time": [1, 2], "event": [1, 2]}, "only 0"),
    ],
)
def test_input_validation(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        kaplan_meier(**kwargs)


def test_hazard_ratio_of_one_is_rejected():
    with pytest.raises(ValueError, match="no effect"):
        events_required(1.0)
