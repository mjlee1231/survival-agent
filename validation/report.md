# Validation report

survagent 0.1.0 · 5000 replicates per check · seed 20261030 · run 2026-10-30T20:32:44+00:00 (31.8 s)

Simulated two-arm trials: exponential survival, control median 12 months, uniform accrual over 24 months, 12 months of further follow-up, 1:1 randomisation. A check passes when the estimate is within 3 Monte Carlo standard errors of its target.

| Check | Target | Estimate | 95% MC interval | Result |
| --- | --- | --- | --- | --- |
| Log-rank rejection rate under H0 (HR = 1), n = 200 | 0.05 | 0.049 | 0.043–0.055 | Pass |
| Coverage of 95% Wald CI for HR = 0.7, n = 200 | 0.95 | 0.956 | 0.950–0.962 | Pass |
| Coverage of 95% log-log CI for S(12) = 0.5, n = 100 | 0.95 | 0.955 | 0.948–0.960 | Pass |
| Simulated power of the design from sample_size (HR = 0.7, N = 370); target 0.80 | 0.80 | 0.797 | 0.786–0.808 | Pass |
| PH-test rejection rate when PH holds (HR = 0.7), n = 400 | 0.05 | 0.046 | 0.041–0.052 | Pass |
| PH-test rejection rate with a delayed effect (HR 1 for 6 months, then 0.5), n = 400 | — | 0.649 | 0.635–0.662 | Info |

## Details

- **Cox estimator:** mean log HR -0.3619 vs true -0.3567 (bias -0.0052); empirical SE 0.1725 vs mean model SE 0.1757.
- **Designed trial:** `sample_size` gave N = 370 for 247 target events; simulations averaged 247.1 events.
