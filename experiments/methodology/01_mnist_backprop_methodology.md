# Experiment 01: MNIST Backpropagation Baseline

## Goal

Establish supervised training baseline on MNIST and verify that consciousness module operates in critical regime (R ≈ 0.41) throughout training. Compare with v3.1.1 reference.

## Methodology

**Model:** SubstanceNet num_classes=10, in_channels=1, Hebbian disabled. Seed=42.

**Training:** 1 epoch, batch_size=64, Adam lr=0.001. TemporalConsciousnessController in 'stream' mode. Metrics logged every 50 batches.

**Test:** Full MNIST test set (10,000 images), per-class accuracy.

**Processing mode:** Image (`mode='image'`). Full V1→V2→V3→V4 pipeline.

## Results

| Metric | v4 (seed=42) | v3.1.1 reference |
|--------|-------------|-----------------|
| Test accuracy | 98.02% | 93.74% |
| Reflexivity R | 0.4099 | 0.382 |
| Phase | critical | critical |

**Training dynamics:**
- Loss: 3.23 → 0.52 (monotonic decrease with noise)
- Accuracy: 42% → 94% (rapid learning in first 200 batches)
- R: 0.4103 → 0.4096 — virtually constant throughout training (range <0.002)

**Per-class accuracy:**
- Best: digit 1 (99.47%), digit 0 (99.08%)
- Worst: digit 8 (95.38%), digit 5 (96.64%)

## Conclusions

1. v4 exceeds v3.1.1 by 4.28% (98.02% vs 93.74%) — BiologicalV1 + V2 pipeline improvement
2. R remains in [0.409, 0.411] from batch 1 to batch 938 — consciousness stable throughout learning
3. No phase transitions during training — system starts and stays in critical regime
4. Per-class pattern: curved digits (8, 5) harder than straight/simple (1, 0) — biologically plausible

## Figures

- `figures/mnist_training.png` — three-panel: loss, accuracy, R dynamics

## Data

- `experiments/results/01_mnist_backprop.json`
