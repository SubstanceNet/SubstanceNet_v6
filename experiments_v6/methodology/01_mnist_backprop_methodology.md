# Experiment 01: MNIST Backpropagation Baseline

## Goal

Establish supervised training baseline on MNIST and verify that consciousness module operates in critical regime (R ≈ 0.41) throughout training.

## Methodology

**Model:** SubstanceNet v6, num_classes=10, plain vector projection (no wave formalism). Hebbian disabled. Seed=42.

**Training:** 1 epoch, batch_size=64, Adam lr=0.001. TemporalConsciousnessController in 'stream' mode. Metrics logged every 50 batches.

**Test:** Full MNIST test set (10,000 images), per-class accuracy.

**Processing mode:** Image (`mode='image'`). Full V1→V2→V3→V4 pipeline with feature_proj.

## Results

| Metric | v6 | v3.1.1 reference |
|--------|-----|-----------------|
| Test accuracy | **97.43%** | 93.74% |
| Reflexivity R | 0.4096 | 0.382 |
| Phase | critical | critical |

**Training dynamics:**
- Loss: 2.13 → 0.04 (monotonic decrease with noise)
- Accuracy: 87.5% → 98.4% (rapid learning)
- R: 0.410 throughout training (range < 0.007)

**Per-class accuracy:** Best: digit 1 (99.3%), digit 2 (99.0%). Worst: digit 4 (93.3%), digit 7 (96.0%).

## Conclusions

1. v6 exceeds v3.1.1 by 3.7% (97.43% vs 93.74%)
2. R remains in [0.406, 0.412] — consciousness stable throughout learning
3. Plain vector projection performs better than old wave formalism (+2.0% vs v5)

## Figures

- `figures/mnist_training.png` — three-panel: loss, accuracy (75-100%), R dynamics
