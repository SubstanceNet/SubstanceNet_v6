<!--
System Classification: project.readme
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Theoretical Framework:
    - 2D-Substance Theory (Onasenko, 2025-2026)
    - The Emergence Parameter κ ≈ 1 (Onasenko, 2025)
    - Reflexive Consciousness Theorem (Th 6.22)
    - Visual Cortex Hierarchy (Hubel & Wiesel, 1962)

Версія: 0.1.0
Дата: 2026-02-11
Мова документації: Українська
Мова коду: English
-->

# SubstanceNet v4

Bio-inspired neural network implementing reflexive consciousness based on 2D-Substance Theory.

## Theoretical Foundation

SubstanceNet implements computational analogues of three key concepts from 2D-Substance Theory:

1. **Wave Functions on Σ** — input features are mapped to complex-valued representations ψ = A·e^(iφ)
2. **Projection Operator P̂** — visual cortex hierarchy V1→V2→V3→V4 (Hubel & Wiesel, 1962)
3. **Reflexive Consciousness** — ψ_C = F[P̂[ψ_C]] (Theorem 6.22) — the system evaluates its own output

The emergence parameter κ ≈ 1 governs the critical regime where optimal performance occurs at sub-maximal reflexivity R ∈ [0.35, 0.47].

## Key Results

| Dataset | Accuracy | Reflexivity | Mode |
|---------|----------|-------------|------|
| MNIST (1 epoch) | 93.74% | 0.382 | Stream |
| MNIST (1 epoch) | 93.47% | 0.468 | Balanced |
| CIFAR-10 | 74.23% | 0.379 | — |

## Architecture
```
Input → QuantumWave → V1 → V2 → V3 → V4 → Reflexive → Output
                                                ↑          |
                                                └──────────┘
                                               (self-evaluation)
```

## Installation
```bash
cd /media/ssd2/ai_projects/SubstanceNet_v4
pip install torch torchvision numpy matplotlib scipy
```

## Quick Start
```python
from src.model import SubstanceNet
from src.constants import CONSCIOUSNESS_MODES

model = SubstanceNet(num_classes=10, mode='stream')
```

## Project Structure
```
SubstanceNet_v4/
├── src/                          # Core modules
│   ├── constants.py              # Single source of truth
│   ├── wave/                     # ψ = A·e^(iφ) wave functions
│   ├── cortex/                   # V1→V2→V3→V4 projection
│   ├── consciousness/            # Reflexive consciousness + controller
│   ├── model/                    # Integrated SubstanceNet
│   └── visualization/            # Dynamics plotting
├── tests/                        # pytest test suite
├── research/                     # Experiments
├── docs/                         # Documentation (Ukrainian)
├── data/                         # Datasets
├── outputs/                      # Results
└── checkpoints/                  # Saved models
```

## References

- Onasenko O. (2026) *Monograph "2D-Substance"*
- Hubel D.H., Wiesel T.N. (1962) *J. Physiol.* 160:106-154
- Yerkes R.M., Dodson J.D. (1908) *J. Comp. Neurol. Psychol.* 18:459-482
- Beggs J.M., Plenz D. (2003) *J. Neurosci.* 23:11167-11177

## License

MIT License — see [LICENSE](LICENSE)
