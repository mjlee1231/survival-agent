"""Check against published results for the Freireich (1963) 6-MP leukemia trial.

Reference values are the standard textbook / R ``survival`` results for this
dataset (e.g. Kleinbaum & Klein, *Survival Analysis*; R ``survdiff`` and ``coxph``).
"""

import pytest

from survagent.cox import cox_regression
from survagent.km import kaplan_meier, survival_at
from survagent.logrank import logrank_test


def test_km_6mp_arm(leukemia):
    d = leukemia[leukemia.treatment == "6-MP"]
    km = kaplan_meier(d.time, d.event)
    expected = {6: 0.857, 7: 0.807, 10: 0.753, 13: 0.690, 16: 0.627, 22: 0.538, 23: 0.448}
    for t, s in expected.items():
        assert survival_at(km, t)["survival"] == pytest.approx(s, abs=5e-4)
    assert km["median"] == 23


def test_km_placebo_median(leukemia):
    d = leukemia[leukemia.treatment == "placebo"]
    assert kaplan_meier(d.time, d.event)["median"] == 8


def test_logrank(leukemia):
    res = logrank_test(leukemia.time, leukemia.event, leukemia.treatment)
    assert res["statistic"] == pytest.approx(16.79, abs=0.01)
    assert res["df"] == 1
    obs = {g["group"]: g["observed"] for g in res["groups"]}
    assert obs == {"6-MP": 9, "placebo": 21}


@pytest.mark.parametrize("ties,coef,se", [("efron", 1.572, 0.412), ("breslow", 1.509, 0.410)])
def test_cox_coefficient(leukemia, ties, coef, se):
    res = cox_regression(leukemia.time, leukemia.event, {"treatment": list(leukemia.treatment)}, ties=ties)
    c = res["coefficients"][0]
    assert c["coef"] == pytest.approx(coef, abs=1e-3)
    assert c["se"] == pytest.approx(se, abs=1e-3)


def test_cox_global_tests_efron(leukemia):
    res = cox_regression(leukemia.time, leukemia.event, {"treatment": list(leukemia.treatment)})
    g = res["global_tests"]
    assert g["likelihood_ratio"]["statistic"] == pytest.approx(16.35, abs=0.01)
    assert g["wald"]["statistic"] == pytest.approx(14.53, abs=0.01)
    assert g["score"]["statistic"] == pytest.approx(17.25, abs=0.01)
