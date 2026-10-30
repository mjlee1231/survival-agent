"""Cross-check against the independent `lifelines` implementation.

Skipped automatically when lifelines is not installed
(``pip install -e ".[dev]"`` installs it).
"""

import numpy as np
import pandas as pd
import pytest

lifelines = pytest.importorskip("lifelines")
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402

from survagent.cox import cox_regression  # noqa: E402
from survagent.km import kaplan_meier  # noqa: E402
from survagent.logrank import logrank_test  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(11)
    n = 300
    grp = rng.choice(["A", "B", "C"], n)
    age = rng.normal(60, 10, n)
    lam = 0.1 * np.exp(0.03 * (age - 60)) * np.select([grp == "B", grp == "C"], [0.7, 1.4], 1.0)
    t = np.round(rng.exponential(1 / lam), 1)  # rounding creates ties
    c = rng.uniform(5, 30, n)
    return pd.DataFrame({"time": np.minimum(t, c) + 0.1, "event": (t <= c).astype(int), "grp": grp, "age": age})


def test_km_matches_lifelines(sim):
    ours = kaplan_meier(sim.time, sim.event)
    kmf = KaplanMeierFitter().fit(sim.time, sim.event)
    ref = kmf.survival_function_.iloc[:, 0]
    for r in ours["table"]:
        assert r["survival"] == pytest.approx(ref.loc[r["time"]], abs=1e-10)


def test_logrank_matches_lifelines(sim):
    ours = logrank_test(sim.time, sim.event, sim.grp)
    ref = multivariate_logrank_test(sim.time, sim.grp, sim.event)
    assert ours["statistic"] == pytest.approx(ref.test_statistic, rel=1e-8)


def test_cox_matches_lifelines(sim):
    ours = cox_regression(sim.time, sim.event, {"age": sim.age.tolist(), "grp": sim.grp.tolist()})
    X = pd.get_dummies(sim[["age", "grp"]], columns=["grp"], drop_first=True, dtype=float)
    cph = CoxPHFitter().fit(pd.concat([sim[["time", "event"]], X], axis=1), "time", "event")  # Efron
    ref = cph.params_
    got = {c["term"]: c["coef"] for c in ours["coefficients"]}
    assert got["age"] == pytest.approx(ref["age"], abs=1e-5)
    assert got["grp=B (vs A)"] == pytest.approx(ref["grp_B"], abs=1e-5)
    assert got["grp=C (vs A)"] == pytest.approx(ref["grp_C"], abs=1e-5)
