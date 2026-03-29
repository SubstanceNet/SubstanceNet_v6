# SubstanceNet v4 — Architecture

**Version:** 0.4.0
**Updated:** 2026-03-18

---

## Data Flow

### Image mode (`mode='image'`)
```
Input [B, C, H, W]
  → RetinalLayer (RGB→4ch: rods+L/M/S cones, if C=3)
  → BiologicalV1 (GaborFilterBank→SimpleCells→ComplexCells→HyperColumns)
  → AdaptiveAvgPool2d(3,3) → flatten → [B, 9, 64]
  → OrientationSelectivity (Conv1d, ×8 orientations) → [B, 9, 512]
  → QuantumWaveFunction (amplitude_fc + phase_fc)
      → amplitude [B, 9, 64], phase [B, 9, 64]
      → cat → features [B, 9, 128]
  → NonLocalInteraction (MultiheadAttention + gate) → [B, 9, 128]
  → MosaicField18 / V2 (thick/thin/pale stripes) → [B, 9, 128]
  → DynamicFormV3 / V3 (spatial gating, static mode) → [B, 9, 128]
  → ObjectFeaturesV4 / V4 (multi-scale attention) → [B, 9, 128]
  → coherence_fc (128→64) → stability_fc (64→64) → Classifier (576→256→10)
  → coherence output → AbstractionLayer (64→3) → ReflexiveConsciousness (3 iters)
```

### Video mode (`mode='video'`)
```
Input [B, T, C, H, W]
  → per-frame: V1 → Orientation → Wave → NonLocal → V2
      → collect V2 sequence [B, T, 9, 128] + amplitude/phase sequences
  → DynamicFormV3 / V3 (temporal mode: phase interference thick×pale)
  → ObjectFeaturesV4 / V4 → coherence → stability → Classifier
  → AbstractionLayer → ReflexiveConsciousness
```

### Cognitive mode (`mode='cognitive'`)
```
Input [B, *] (flat tensor)
  → pad/truncate to 64 → cognitive_input (Linear(64,576)+ReLU)
  → reshape [B, 9, 64]
  → OrientationSelectivity → Wave → NonLocal → V2 → V3 → V4 → ...
```

---

## Module Reference

### Visual Cortex (src/cortex/)

| Module | File | Function | Innate? | Params |
|--------|------|----------|---------|--------|
| RetinalLayer | v1.py | RGB → 4 photoreceptors (rods+cones) | Yes (fixed) | 0 |
| BiologicalV1 | v1.py | Gabor→Simple→Complex→Hyper | Yes (fixed filters) | 141K |
| MosaicField18 | v2.py | thick(roll-diff) / thin(FFT) / pale(pass) | Yes (parameter-free) | 0 |
| DynamicFormV3 | v3.py | Phase interference + HebbianLinear | Partially (Hebbian) | 52K |
| ObjectFeaturesV4 | v4.py | Multi-scale attention + HebbianLinear | No (Hebbian) | 49K |
| HebbianLinear | hebbian.py | ΔW = η·(coherence·x·y − α·W·y²) | N/A | varies |

### Wave (src/wave/)

| Module | File | Function | Params |
|--------|------|----------|--------|
| QuantumWaveFunction | quantum_wave.py | features → amplitude A, phase φ; ψ = A·e^(iφ) | 66K |

### Consciousness (src/consciousness/)

| Module | File | Function | Params |
|--------|------|----------|--------|
| ReflexiveConsciousness | reflexive.py | ψ_C = F[P̂[ψ_C]], 3 iterations, R-targeting | 6K |
| TemporalController | controller.py | Phase monitoring: subcritical/critical/supercritical/saturated | 0 |

### Memory (src/hippocampus/)

| Module | File | Function | Params |
|--------|------|----------|--------|
| GridCells | cells.py | Hexagonal spatial encoding | — |
| PlaceCells | cells.py | Gaussian receptive fields | — |
| TimeCells | cells.py | Logarithmic temporal scales | — |
| EpisodicEncoder | episodic_memory.py | Episode formation from context | — |
| ConsciousRetrieval | episodic_memory.py | Similarity search modulated by ψ_C | — |
| MemoryConsolidation | episodic_memory.py | Short-term → prototype compression | — |
| Hippocampus | hippocampus.py | Integrated episodic memory system | 757K |

### Model (src/model/)

| Module | File | Function | Params |
|--------|------|----------|--------|
| SubstanceNet | substance_net.py | Main model: 3 modes, loss, metrics | 1.36M |
| OrientationSelectivity | layers.py | Conv1d ×8 orientations | 2K |
| NonLocalInteraction | layers.py | MultiheadAttention + learnable gate | 66K |
| AbstractionLayer | layers.py | mean→Linear→ReLU→Linear (dim→3) | 2K |

---

## Key Metrics

| Metric | Meaning | Optimal |
|--------|---------|---------|
| R (reflexivity) | Convergence of ψ_C = F[P̂[ψ_C]] | 0.35–0.47 |
| κ (emergence) | Critical regime indicator | ≈ 1.0 |
| Coherence | Phase alignment across positions | High (>0.99) |
| Abstract variance | Diversity of abstraction layer output | >10 (not collapsed) |

### R interpretation

| R range | Phase | Interpretation |
|---------|-------|---------------|
| < 0.30 | Subcritical | Insufficient self-monitoring |
| 0.30–0.50 | **Critical** | Optimal operating regime (κ ≈ 1) |
| 0.50–0.80 | Supercritical | Excessive synchronization |
| > 0.80 | Saturated | Representation collapse |

---

## Loss Function
```
L_total = L_classification
        + 0.5 · L_abstract
        + 0.1 · L_consciousness (R-targeting: (MSE − 1.44)²)
        + 0.01 · L_phase_coherence
        + 0.01 · L_topological
        + L_r_penalty (when R outside [0.35, 0.47])
```

---

## Hebbian Learning

V3 and V4 use HebbianLinear instead of nn.Linear for upper cortical layers.
Weights update during forward pass, not backward:
```
ΔW_ij = η · (cos(φ_i − φ_j) · x_i · y_j − α · W_ij · y_j²)
```

- First term (Hebb): strengthen where phases cohere
- Second term (Oja): normalize to prevent unbounded growth
- `requires_grad=False` — no backpropagation for these weights
- V3 learning_rate=0.0001, oja_alpha=0.1
- V4 learning_rate=0.0001, oja_alpha=1.0

---

## Hippocampus API
```python
# Store episode (after forward + compute_loss)
model.store_episode(output, task_type='logic', metrics={'accuracy': 0.95})

# Recall similar episodes
similar = model.recall(output, top_k=5)

# Consolidate (short-term buffer → prototypes)
model.consolidate_memory()
```

Hippocampus operates parallel to the forward pass, not inline.
Consciousness amplitude modulates storage importance.

---

## File Dependencies
```
substance_net.py
├── cortex/v1.py        (BiologicalV1, RetinalLayer)
├── cortex/v2.py        (MosaicField18)
├── cortex/v3.py        (DynamicFormV3)
│   └── cortex/hebbian.py (HebbianLinear)
├── cortex/v4.py        (ObjectFeaturesV4)
│   └── cortex/hebbian.py (HebbianLinear)
├── wave/quantum_wave.py (QuantumWaveFunction)
├── consciousness/reflexive.py (ReflexiveConsciousness)
├── model/layers.py     (OrientationSelectivity, NonLocal, Abstraction)
├── hippocampus/hippocampus.py
│   ├── hippocampus/episodic_memory.py
│   └── hippocampus/cells.py
└── constants.py

consciousness/controller.py  (external, used by training scripts)
data/dynamic_primitives.py   (video data generator)
utils.py                     (cognitive task generators)
```
