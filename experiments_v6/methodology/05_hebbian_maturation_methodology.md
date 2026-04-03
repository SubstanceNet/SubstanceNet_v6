# Experiment 05: Hebbian Maturation

## Goal

Demonstrate unsupervised V3/V4 weight adaptation through observation of dynamic stimuli, without labels or backpropagation. Verify that relevant experience improves recognition more than irrelevant.

## Methodology

**Maturation:** 500 steps observing random dynamic primitives (4 shapes, random speeds 0.5-4.0, random directions). HebbianLinear with Oja normalization.

**Measurements:** V3 motion response (speed=2.0), shape discrimination, 20-shot kNN recognition.

**Three conditions:** Fresh (random weights), matured on primitives (irrelevant to MNIST), matured on MNIST images (relevant).

## Results

| Metric | Fresh | Matured (primitives) | Matured (MNIST) |
|--------|-------|---------------------|-----------------|
| V3 response (speed=2.0) | 3.62 | 5.85 (1.6×) | — |
| Recognition 20-shot | 61.9% | 62.4% (+0.5%) | **64.4% (+2.4%)** |
| Weight norm | 1.0 | **35.6** (Oja stable) | — |

Hebbian maturation amplifies V3 motion response by 1.6×. Weight norm stabilizes at ~35.6 (Oja normalization prevents divergence). Recognition improvement is modest but shows clear pattern: relevant data (+2.4%) > irrelevant data (+0.5%).

## Conclusions

1. V3 motion signal amplified 1.6× after unsupervised maturation
2. Oja normalization stabilizes weights (norm → 35.6)
3. Relevant experience (MNIST) helps more than irrelevant (primitives)
4. Biologically plausible: sensitive period effect

## Figures

- `figures/hebbian_maturation.png` — amplification, Oja norm, velocity before/after
- `figures/hebbian_recognition.png` — recognition comparison (3 conditions)
