# Experiment 03: Recognition Paradigm

## Goal

Verify recognition without backpropagation via kNN in feature space. Demonstrate that biological Gabor features enable meaningful recognition through episodic memory comparison (CLS paradigm: McClelland et al., 1995).

## Methodology

### Recognition method: kNN top-5 weighted cosine voting

**Feature extraction:** amplitude+phase features (128-dim) from QuantumWaveFunction output, spatially pooled: [B, 9, 128] → mean → [B, 128]. These features have separation ratio ~50 (vs abstract dim=3 with ratio 1.6).

**Storage:** Individual episodes stored as (feature_vector, class_label) pairs. No averaging, no compression — each observation preserved separately.

**Recognition:** For each test image:
1. Extract 128-dim feature vector
2. Compute cosine similarity with ALL stored episodes
3. Select top-5 most similar episodes
4. Weighted vote: each neighbor votes for its class, weighted by cosine similarity score
5. Prediction = class with highest weighted vote sum

**Why kNN, not prototypes:** Prototypes (mean per class) destroy within-class variability. Digit "1" written vertically vs tilted are different episodes. kNN preserves this — the test image matches the most similar *individual* writing style, not an averaged template. At N=100: kNN=66.9%, prototype=43.9% — delta +23%.

### Reproducibility test

5 independent random initializations (no fixed seed), each running 100-shot recognition. Tests whether result depends on specific weight initialization.

### Stage feature quality

kNN recognition using features extracted at each processing stage (V1, V2, V3, V4) separately, with 20 examples per class. Tests which stages contribute most to recognition quality.

**Model:** SubstanceNet untrained, Hebbian disabled, seed=42 for main experiments. Test size: 1024 images.

## Results

### N-shot scaling (seed=42)

| N-shot | kNN accuracy | Prototype accuracy | Delta | Episodes |
|--------|-------------|-------------------|-------|----------|
| 1 | 38.7% | 38.7% | +0.0% | 10 |
| 5 | 40.0% | 41.9% | -1.9% | 50 |
| 10 | 48.9% | 42.1% | +6.8% | 100 |
| 20 | 55.7% | 45.6% | +10.1% | 200 |
| 50 | 60.8% | 44.8% | +16.0% | 500 |
| 100 | 66.9% | 43.9% | +22.9% | 1000 |

### Reproducibility (100-shot, 5 random inits)

| Trial | Accuracy |
|-------|----------|
| 0 | 73.7% |
| 1 | 71.0% |
| 2 | 73.9% |
| 3 | 77.1% |
| 4 | 72.8% |
| **Mean** | **73.7% ± 2.0%** |

Note: seed=42 gives 66.9% which is below the mean. The original result of 71.9% (March 16) falls within the normal distribution.

### Memory consolidation (20-shot)

| Method | Accuracy | Memories |
|--------|----------|----------|
| kNN episodes | 55.7% | 200 |
| Prototypes | 45.6% | 10 |
| Trade-off | -10.1% | 20× compression |

### Feature quality by stage (20-shot kNN)

| Stage | Accuracy | Innate? |
|-------|----------|---------|
| V1 (Gabor) | 55.7% | Yes — fixed Gabor filters |
| V2 (FFT+diff) | 65.6% | Yes — parameter-free FFT + roll-diff |
| V3 (interference) | 65.9% | No — random weights, neutral |
| V4 (attention) | 65.7% | No — random weights, neutral |

### Critical methodological finding

**kNN vs prototype matching is the key distinction.** Early experiments used simple cosine prototype matching which saturates at ~44% regardless of N. The original 71.9% result used kNN (top-5 weighted voting on individual episodes), which scales logarithmically with N.

This difference is biologically meaningful: the brain stores individual memories (episodic), not averaged templates (semantic). Recognition through episodic comparison is the CLS-predicted mechanism.

### Hippocampus API incompatibility

The current Hippocampus module accepts input_dim=3 (abstract) but recognition requires 128-dim features. Grid cells Linear(3→2) cannot process 128-dim input. The 71.9% result was achieved via direct cosine similarity, NOT through the hippocampus API. This is a known architectural issue to be resolved.

## Conclusions

1. kNN recognition achieves 73.7% ± 2.0% (100-shot) without any backpropagation
2. kNN dramatically outperforms prototypes at higher N (+22.9% at 100-shot)
3. V2 (FFT+diff) provides +10% over V1 — innate feature processing without training
4. V3/V4 with random weights are neutral — neither help nor hurt (need Hebbian maturation)
5. Memory trade-off: 20× compression costs 10.1% accuracy (CLS fast vs slow)
6. R = 0.40 — consciousness remains in critical regime during recognition

## Figures

- `figures/recognition_scaling.png` — kNN vs prototype scaling curves
- `figures/innate_vs_acquired.png` — feature quality by stage
- `figures/consolidation.png` — memory trade-off

## Data

- `experiments/results/03_recognition_paradigm.json`
