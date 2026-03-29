# Experiment 05: Hebbian Maturation

## Goal

Verify that Hebbian plasticity (phase-coherence learning with Oja normalization) enables unsupervised V3/V4 weight adaptation through observation. Test whether maturation on relevant vs irrelevant stimuli differentially affects recognition — matching biological sensitive periods.

## Methodology

**Model:** SubstanceNet num_classes=10, in_channels=1. Seed=42 for fresh model initialization. Hebbian learning enabled via `enable_hebbian()` + `model.train()` (HebbianLinear updates only in training mode).

**Hebbian update rule:**
```
ΔW_ij = η · (cos(φ_i − φ_j) · x_i · y_j − α · W_ij · y_j²)
```
V3: lr=0.0001, oja_alpha=0.1. V4: lr=0.0001, oja_alpha=1.0.

**Maturation protocol:** 500 forward passes without labels or backprop. Random primitive type (circle/square/triangle/line), random speed (0.5–4.0), random direction. Every 10 steps: measure V3 response at speed=2.0 and weight norm.

**Three conditions tested:**
1. Fresh model (random weights, no maturation)
2. Matured on primitives (500 steps, geometric shapes)
3. Matured on MNIST (500 steps, actual digits)

**Recognition test:** kNN top-5 weighted cosine voting on 128-dim amplitude+phase features, 20-shot, 1024 test images. Model set to `eval()` before recognition to stop Hebbian updates during testing.

**Velocity measurement:** V3 output hook, L2 norm difference between moving (speed=2.0) and static reference. Speeds tested: 0.0, 0.5, 1.0, 2.0, 3.0, 4.0.

**Critical implementation detail:** HebbianLinear checks `self.learning and self.training` — both must be True for weight updates. `model.train()` required during maturation, `model.eval()` required during testing.

## Results

### Motion signal amplification

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| V3 response (speed=2.0) | 0.170 | 1.373 | **8.1× gain** |
| Shape: line vs closed | 0.039 | 0.394 | 10× gain |
| Shape: closed vs closed | 0.018 | 0.226 | 13× gain |
| Weight norm | 1.56 | ~35.5 | Oja-stabilized |

V3 response amplifies rapidly in first 50 steps, then stabilizes. Weight norm grows from 1.6 to ~35.5 and is held stable by Oja normalization — no divergence.

### Velocity tuning preservation

| Speed | Before | After |
|-------|--------|-------|
| 0.0 | 0.000 | 0.000 |
| 0.5 | 0.070 | 0.563 |
| 1.0 | 0.118 | 0.984 |
| 2.0 | 0.170 | 1.373 |
| 3.0 | 0.209 | 1.616 |
| 4.0 | 0.199 | 1.590 |

Logarithmic saturation shape preserved after maturation. Zero response for static stimuli maintained.

### Recognition comparison

| Condition | 20-shot kNN | Delta from fresh |
|-----------|------------|-----------------|
| Fresh (random) | 55.7% | baseline |
| Matured (primitives) | 58.9% | **+3.2%** |
| Matured (MNIST) | 60.8% | **+5.2%** |

Both types of maturation improve recognition. MNIST maturation helps more (+5.2%) than primitive maturation (+3.2%) — relevant experience is more beneficial, matching biological sensitive period findings (Hubel & Wiesel, 1968).

## Conclusions

1. Hebbian learning produces 8× motion signal amplification without any supervision
2. Oja normalization prevents weight divergence (stabilizes ~35.5)
3. Velocity tuning curve shape preserved after maturation (logarithmic saturation)
4. Relevant stimuli (MNIST) help more than irrelevant (primitives): +5.2% vs +3.2%
5. Both types of maturation improve recognition — general visual experience has value
6. Biological parallel: sensitive period — early visual experience shapes feature detectors

## Figures

- `figures/hebbian_maturation.png` — 3-panel: amplification, Oja stability, velocity before/after
- `figures/hebbian_recognition.png` — recognition comparison across conditions

## Data

- `experiments/results/05_hebbian_maturation.json`
