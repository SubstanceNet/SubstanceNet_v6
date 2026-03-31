# Experiment 09: Wave Formalism Ablation (CRITICAL)

## Goal

Determine whether wave formalism (ψ = A·e^(iφ)) provides computational utility beyond notation. Three-way comparison: Classic (old QuantumWaveFunction), WaveFunctionOnT (new, on configuration space T = 2^i−1), Plain (two Linear layers, no wave).

## Methodology

**Three models compared:**

| Model | Wave | V3 | Params |
|-------|------|----|--------|
| Classic | softplus(Linear) + cos/sin | Phase interference | 1,357,785 |
| WaveOnT | WaveFunctionOnT (7 cliques, Hamming kernel) | Same as Classic | 1,438,728 |
| Plain | ReLU(Linear) + Linear (no trig) | Same V3 | 1,357,785 |

**Seed:** 42 for all. Hebbian disabled for fair comparison.

**Phase 1 — Untrained (random weights):**
- Velocity tuning curve (7 speeds, V3 hook)
- Shape discrimination (4 primitives, speed=2.0)
- Recognition 20-shot kNN (1024 test images)

**Phase 2 — Trained (1 epoch MNIST):**
- MNIST backprop accuracy
- Recognition 20-shot kNN after training
- Velocity curve after training
- Cognitive task (logic, 50 epochs)

## Results

### Phase 1: Untrained

All three models produce identical velocity curve SHAPE (logarithmic saturation).
Old wave = trivially equivalent to plain vectors (confirmed).

| Metric | Classic | WaveOnT | Plain |
|--------|---------|---------|-------|
| Velocity peak | 0.209 | 0.083 | 3.857 |
| Topological ratio | 2.2× | 2.3× | 2.5× |
| Recognition 20-shot | 55.7% | 55.7% | 59.1% |

### Phase 2: Trained

| Metric | Classic | WaveOnT | Plain |
|--------|---------|---------|-------|
| MNIST backprop | 97.5% | 96.7% | 97.3% |
| **Recognition 20-shot** | 78.1% | 79.4% | **85.3%** |
| Cognitive accuracy | 100% | 100% | 100% |
| Cognitive R | 0.412 | 0.411 | 0.409 |

### Key finding

Wave formalism as feature extractor (between V2-V3) does not outperform plain vectors. Plain actually wins on recognition (+7.1% vs Classic). This is because ReLU produces sharper, more discriminative features for kNN.

However, the wave formalism has value in **ensemble dynamics** (within ReflexiveConsciousness), not as a feature transform. Phase interference and nonlocal potential are relevant for binding and synchronization, not for feature extraction.

## Conclusions

1. Old QuantumWaveFunction (softplus+cos/sin) is trivial — confirmed by ablation
2. WaveOnT (7 cliques, Hamming kernel) is structurally correct but not superior as feature extractor
3. Plain vectors are optimal for discriminative features (recognition)
4. Wave formalism's computational utility lies in ensemble dynamics (consciousness), not feature extraction
5. Velocity tuning curve shape (logarithmic saturation) is property of V3 architecture, not wave formalism

## Figures

- `figures/wave_ablation.png` — 4-panel: velocity untrained/trained, recognition comparison, normalized metrics

## Data

- `experiments/results/09_wave_ablation.json`
