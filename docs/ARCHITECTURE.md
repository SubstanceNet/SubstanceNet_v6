# SubstanceNet v6 — Architecture

**Version:** 0.6.0
**Updated:** 2026-04-01

---

## Design Principles

1. **Biological correspondence:** Each module maps to a brain structure
2. **Vectors for features:** V1→V4 use plain vectors (Linear, Conv, ReLU) — matches neuronal spike coding
3. **Wave formalism for binding:** Reserved for inter-modal synchronization (future v7+)
4. **Consciousness as constraint:** R-targeting models physiological limits, not computation
5. **Modular extensibility:** Architecture ready for auditory (A1→A3) and speech (M1→M2) modules

---

## Data Flow

### Image mode (`mode='image'`)
```
Input [B, C, H, W]
  → RetinalLayer (RGB → 4ch: rods + L/M/S cones, if C=3)
  → BiologicalV1 (GaborFilterBank → Simple → Complex → HyperColumns)
  → AdaptiveAvgPool2d(3,3) → flatten → [B, 9, 64]
  → OrientationSelectivity (Conv1d, ×8 orientations) → [B, 9, 512]
  → FeatureProjection (Linear + ReLU) → [B, 9, 128]
      → split: amplitude [B, 9, 64], phase [B, 9, 64]
  → NonLocalInteraction (MultiheadAttention + gate) → [B, 9, 128]
  → MosaicField18 / V2 (thick/thin/pale stripes) → [B, 9, 128]
  → DynamicFormV3 / V3 (spatial gating) → [B, 9, 128]
  → ObjectFeaturesV4 / V4 (multi-scale attention + Hebbian) → [B, 9, 128]
  ├→ coherence_fc (128→64) → stability_fc (64→64)
  │   → Classifier (576→256→num_classes) → logits
  └→ coherence_fc → AbstractionLayer (64→3) → abstract
      → ReflexiveConsciousness (3 iterations) → ψ_C, amplitude_c, phase_c
      → Top-down gate (ψ_C → V4, inactive in v6) ← infrastructure for v7
```

### Video mode (`mode='video'`)
```
Input [B, T, C, H, W]
  → per-frame: V1 → Orientation → FeatureProj → NonLocal → V2
      → collect: V2 [B, T, 9, 128] + amplitude/phase sequences
  → DynamicFormV3 / V3 (temporal mode: phase interference thick×pale)
  → V4 → coherence → stability → Classifier
  → AbstractionLayer → Consciousness
```

### Cognitive mode (`mode='cognitive'`)
```
Input [B, *] (flat tensor)
  → pad/truncate to 64 → cognitive_input (Linear(64, 576) + ReLU)
  → reshape [B, 9, 64]
  → Orientation → FeatureProj → NonLocal → V2 → V3 → V4 → ...
```

---

## Module Reference

### Visual Cortex (`src/cortex/`)

| Module | File | Biological Basis | Parameters |
|--------|------|-----------------|------------|
| BiologicalV1 | v1.py | Hubel & Wiesel (1962) | GaborFilterBank (fixed) + Simple/Complex/Hyper |
| MosaicField18 (V2) | v2.py | Livingstone & Hubel (1987) | Thick/thin/pale stripes, FFT+diff |
| DynamicFormV3 | v3.py | Felleman & Van Essen (1991) | Cross-stream gating, temporal diff |
| ObjectFeaturesV4 | v4.py | Zeki (1983), Pasupathy (2001) | Multi-scale attention + Hebbian |
| HebbianLinear | hebbian.py | Hebb (1949), Oja (1982) | Oja-normalized weight updates |

### Consciousness (`src/consciousness/`)

| Module | File | Theory | Role |
|--------|------|--------|------|
| ReflexiveConsciousness | reflexive.py | ψ_C = F[P̂[ψ_C]] | R-targeting, regularization |
| TemporalController | controller.py | κ ≈ 1 (Onasenko, 2025) | Phase monitoring, inertia |

**Consciousness data flow:**
```
abstract [B, 3] → psi_c_init → iterate 3× {
    P̂: project → threshold → LayerNorm
    F: cat(projected, abstract) → Linear+Tanh+Linear
    mix: α·new + (1-α)·old
} → amplitude_c [B, 16], phase_c [B, 16]
→ consciousness_loss (R-targeting, coherence, stability, entropy)
→ top-down gate (inactive): sigmoid(Linear(amplitude_c)) → V4 modulation
```

R-targeting: R = 1/(1+MSE(ψ_C, P̂[ψ_C])) → target_mse=1.44 → R ≈ 0.41
This models physiological constraints (GABA/glutamate balance, metabolic budget).

### Hippocampus (`src/hippocampus/`)

| Module | File | Biological Basis |
|--------|------|-----------------|
| GridCells | cells.py | Moser et al. (2005) |
| PlaceCells | cells.py | O'Keefe (1976) |
| EpisodicEncoder | episodic_memory.py | Tulving (1972) |
| ConsciousRetrieval | episodic_memory.py | Consciousness-modulated recall |
| MemoryConsolidation | episodic_memory.py | McClelland et al. (1995) |
| Hippocampus | hippocampus.py | Complete memory system |

**Two memory interfaces:**
- `store_episode(abstract)` → episodic memory (dim=3, contextual)
- `store_feature(features, label)` → recognition memory (dim=128, discriminative)
- `recognize(features)` → kNN on encoded feature memory

### Other Modules

| Module | File | Role |
|--------|------|------|
| FeatureProjection | model/substance_net.py | Linear(512→128) + ReLU, replaces wave |
| NonLocalInteraction | model/layers.py | MultiheadAttention + sigmoid gate |
| OrientationSelectivity | model/layers.py | Conv1d ×8 orientations |
| AbstractionLayer | model/layers.py | Linear(64→32→3) + dropout |
| RetinalLayer | model/layers.py | RGB→rods+L/M/S cones |

---

## Parameters

Total: **1,458,775** (v6)

| Module | Count | % |
|--------|-------|---|
| V1 (BiologicalV1) | ~50K | 3.4% |
| Orientation | ~33K | 2.3% |
| FeatureProjection | ~66K | 4.5% |
| NonLocal | ~100K | 6.9% |
| V2 (MosaicField18) | ~200K | 13.7% |
| V3 (DynamicFormV3) | ~130K | 8.9% |
| V4 (ObjectFeaturesV4) | ~50K | 3.4% |
| Classifier | ~160K | 11.0% |
| Consciousness | ~5K | 0.3% |
| Top-down gate | ~2K | 0.1% |
| Hippocampus | ~660K | 45.3% |

---

## Infrastructure for v7

### Top-down modulation (inactive in v6)
```python
self.topdown_gate = nn.Sequential(
    nn.Linear(consciousness_dim // 2, feature_dim),
    nn.Sigmoid())
# Initialized to zero weights → gate = 0.5 → neutral
# v4_features *= (1 + topdown_weight * (gate - 0.5))
# topdown_weight = 0 → no effect
```

Activation: set `topdown_weight > 0` and train. Biological basis: prefrontal → V4 feedback.

### Wave dynamics (in research/)

WaveFunctionOnT and ReflexiveConsciousnessV2 preserved in `research/wave_dynamics/` for future inter-modal binding experiments. Not used in v6 pipeline.

### Modality interface (future)

Architecture supports plug-and-play modules:
```
Layer 1 (sensory):  V1→V4 (visual), A1→A3 (auditory), M1→M2 (speech)
Layer 2 (memory):   Hippocampus (multimodal, feature_dim=128 per modality)
Layer 3 (binding):  WaveFunctionOnT (wave resonance, future)
Layer 4 (consciousness): ψ_C = F[P̂[ψ_C]] with top-down to V4
```

---

## Key Results (v6, seed=42)

| Experiment | Result | Reference |
|------------|--------|-----------|
| MNIST backprop | 97.4% | v3.1.1: 93.7% |
| Cognitive R | 0.4090 ± 0.0009 | κ-plateau |
| Recognition 100-shot | 73.2% ± 2.1% | No backprop |
| Velocity peak | 4.64 | Logarithmic saturation |
| Hebbian amplification | 1.6× | +2.4% recognition |
| κ ≈ 1 | 0.993 ± 0.010 | He-II: 0.989 ± 0.007 |
