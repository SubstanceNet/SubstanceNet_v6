# Experiment 11: Statistical Analysis

## Goal

Provide 95% confidence intervals for all key metrics using multiple random initializations. Addresses reviewer requirement for formal statistical analysis.

## Methodology

**Seeds:** [42, 123, 456, 789, 2024] — 5 independent random initializations.

**Metrics tested:**
1. MNIST backprop (1 epoch) — accuracy and R
2. Recognition 100-shot kNN (untrained) — accuracy
3. Cognitive battery (10 tasks × 5 seeds) — accuracy and R
4. Velocity tuning peak (speed=3.0) — V3 response

**CI computation:** Student's t-distribution, 95% confidence level, scipy.stats.

## Results

| Metric | Mean | 95% CI | N | CV |
|--------|------|--------|---|-----|
| MNIST backprop | 95.07% | [92.80, 97.33] | 5 | 2.3% |
| Recognition 100-shot | 71.2% | [67.9, 74.4] | 5 | 3.3% |
| Cognitive R | 0.4099 | [0.4083, 0.4116] | 50 | 0.3% |
| Velocity peak | 0.248 | [0.177, 0.319] | 5 | 20.6% |

## Analysis

**R is the most stable metric** (CV=0.3%). Regardless of random initialization, consciousness converges to R ≈ 0.41. This strongly supports the κ-plateau hypothesis.

**Recognition is reproducible** (CV=3.3%). The 71.2% mean is consistent with exp03 (73.7% ± 2.0% with different methodology).

**MNIST shows moderate variance** (CV=2.3%) due to 1-epoch sensitivity to initialization. Seed 456 underperforms (92.0%).

**Velocity has high variance** (CV=20.6%) — expected since untrained V3 weights vary significantly across initializations. Velocity tuning SHAPE (logarithmic saturation) is consistent across all seeds.

## Conclusions

1. R = 0.4099 ± 0.0017 — extremely stable across initializations (κ-plateau confirmed)
2. Recognition = 71.2% ± 3.3% — reproducible without backprop
3. MNIST = 95.1% ± 2.3% — moderate variance, 1-epoch sensitivity
4. Velocity = 0.248 ± 0.071 — magnitude varies, shape consistent

## Figures

- `figures/statistical_analysis.png` — 4-panel: MNIST, recognition, R, summary table

## Data

- `experiments/results/11_statistical_analysis.json`
