# Experiment 02: Cognitive Task Battery

## Goal

Verify κ-plateau: reflexivity R ≈ 0.41 across 10 different cognitive tasks, demonstrating task-independent critical regime.

## Methodology

**Tasks:** logic, memory, categorization, analogy, spatial, raven, numerical, verbal, emotional, insight.

**Training:** 50 epochs per task, batch_size=32, Adam lr=0.001. num_classes=2 (binary). Seed=42.

**Evaluation:** 256 test samples per task. Metrics: accuracy, R, phase coherence, abstract variance.

## Results

| Metric | Value |
|--------|-------|
| Mean accuracy | **99.84%** |
| Mean R | **0.4090 ± 0.0009** |
| All in critical regime | Yes (10/10 tasks) |
| Abstract variance range | 8.4 (raven) — 25.3 (verbal) |

All 10 tasks converge to R ≈ 0.41 regardless of task type. Abstract variance differentiates task complexity: raven (hardest, lowest variance) vs verbal (highest variance).

## Conclusions

1. κ-plateau confirmed: R = 0.4090 ± 0.0009 across all 10 tasks
2. System operates in critical regime independent of task type
3. Abstract representation differentiates task complexity (3× range)

## Figures

- `figures/kappa_plateau.png` — R per task + abstract variance
- `figures/kappa_convergence.png` — R during training for all tasks
