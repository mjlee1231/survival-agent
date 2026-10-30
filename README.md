# survagent — validated survival-analysis tools for AI agents

Clinical-trial statistics (Kaplan–Meier, log-rank, Cox regression, sample size and
power) packaged as tools an AI agent can call through the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io). The focus is on
whether an agent's answers can be trusted:

- **Uncertainty is part of every answer.** Results come with confidence intervals,
  and with assumption checks (proportional hazards, events per covariate, small
  expected counts, medians not reached) returned as plain-language `notes`.
- **Every number is traceable.** Datasets are registered once and identified by a
  SHA-256 hash. Each result carries a provenance record (function, inputs, dataset
  hash, package version, timestamp), so it can be reproduced.
- **The tools are validated.** They are checked against published results, an
  independent reference implementation, and a simulation study of their operating
  characteristics.

## Validation

| Layer | What it checks | Where |
| --- | --- | --- |
| Published results | KM estimates, log-rank χ², Cox coefficients (Efron and Breslow) and LR/Wald/score tests on the Freireich 6-MP leukemia trial match textbook / R `survival` values | `tests/test_published_values.py` |
| Reference implementation | KM, multi-group log-rank and multivariable Cox with ties agree with `lifelines` | `tests/test_reference_lifelines.py` |
| Mathematical properties | Cox score test = log-rank without ties; invariance to row order; sign flip under a change of reference level; power formula inverts the events formula | `tests/test_properties.py` |
| Operating characteristics | Type I error, CI coverage, achieved power of a designed trial, PH-test size and power (5,000 simulated trials per check) | `validation/report.md` |

Latest simulation study (5,000 replicates per check; details in
[`validation/report.md`](validation/report.md)):

| Check | Target | Estimate |
| --- | --- | --- |
| Log-rank type I error | 0.05 | 0.049 |
| Cox 95% CI coverage (HR = 0.7) | 0.95 | 0.956 |
| KM 95% log-log CI coverage | 0.95 | 0.955 |
| Power of the trial designed by `sample_size` | 0.80 | 0.797 |
| PH-test size when PH holds | 0.05 | 0.046 |
| PH-test detection of a 6-month delayed effect | — | 0.649 |

## Setup

Requires Python 3.10+.

```bash
cd survival-agent
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # dev adds pytest and lifelines (reference checks)
pytest                             # unit, reference and MCP tests
python validation/run_validation.py --reps 500   # quick simulation study
```

## Using it from an AI agent

### Claude Desktop (no API key needed)

Add the server to Claude Desktop's config file
(`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS),
then restart Claude Desktop:

```json
{
  "mcpServers": {
    "survagent": {
      "command": "/ABSOLUTE/PATH/TO/survival-agent/.venv/bin/survagent-mcp"
    }
  }
}
```

Then ask, for example: *"Load /ABSOLUTE/PATH/TO/survival-agent/data/leukemia_6mp.csv,
compare survival between treatment arms, and fit a Cox model. Report hazard ratios
with confidence intervals and any assumption warnings."*

### Tools

| Tool | Purpose |
| --- | --- |
| `load_dataset` | Register a CSV; returns id, SHA-256 hash and a column summary |
| `kaplan_meier` | KM curves (overall or by group), log-log CIs, median survival with CI |
| `logrank_test` | Log-rank test for two or more groups |
| `cox_regression` | Hazard ratios with CIs, LR/Wald/score tests, proportional-hazards test |
| `events_required` | Events for a two-sided log-rank test (Schoenfeld) |
| `power_from_events` | Power for a given number of events |
| `sample_size` | Total N under exponential survival and uniform accrual |
| `simulate_power` | Monte Carlo check of a design's power, with a Monte Carlo CI |

## Methods

- **Kaplan–Meier:** Greenwood variance; log(−log S) confidence intervals; median CI
  by inverting the pointwise bands (as in R `survfit`).
- **Log-rank:** Mantel–Haenszel statistic with hypergeometric variance, k groups.
- **Cox:** Newton–Raphson with step halving; Efron (default) or Breslow ties; Wald
  CIs; Grambsch–Therneau proportional-hazards test on scaled Schoenfeld residuals
  against 1 − KM(t).
- **Design:** Schoenfeld (1981) events formula; event probability under exponential
  survival with uniform accrual; simulation-based power.

The methods are implemented directly in NumPy/SciPy so every step is inspectable,
and they are cross-checked against `lifelines`.

## Project layout

```
src/survagent/     statistical methods, tool layer, MCP server
tests/             published-value, reference, property and MCP tests
validation/        simulation study and its latest report
data/              Freireich (1963) 6-MP leukemia trial (42 patients)
```

## Roadmap

1. **Validated statistical tools behind MCP** — done (this release).
2. **Evidence retrieval (RAG):** index regulatory guidance (ICH E9, E9(R1)) and
   PubMed abstracts on survival methods; hybrid retrieval (BM25 + embeddings).
   Evaluate retrieval on a labelled question set (recall@k, MRR) with bootstrap CIs,
   and compare retrieval strategies on the same questions.
3. **Agent and end-to-end evaluation:** an agent that retrieves guidance, calls the
   statistical tools and drafts an interpretation for human review, citing both
   sources and provenance. Evaluate answer faithfulness, tool-call correctness and
   failure modes on a fixed benchmark of trial-analysis questions.

## Data

`data/leukemia_6mp.csv`: remission durations (weeks) from the Freireich et al.
(1963, *Blood* 21:699–716) trial of 6-mercaptopurine versus placebo in acute
leukemia, as tabulated in Gehan (1965) and many survival-analysis textbooks.
`event` = 1 for relapse, 0 for censored.
