# Experiment 04: Velocity Tuning Curve

## Goal

Verify that V3 produces velocity-dependent response matching primate MT/V3 electrophysiology: zero response for static stimuli, logarithmic saturation, and rolloff at high speeds.

## Methodology

**Stimuli:** Dynamic primitives (circle), 9 speeds: {0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0} px/frame. 6 frames per sequence.

**Measurement:** L2 norm difference between V3 output for moving vs static stimulus (V3 hook).

**Shape discrimination:** 4 primitives (circle, square, triangle, line) at speed=2.0.

**Translation invariance:** V3 raw diff vs V4 abstract diff at speed=3.0.

## Results

| Metric | Value |
|--------|-------|
| Peak response | **4.64** at speed=3.0 |
| Static response | 0.0 (zero phantom motion) |
| Saturation | Logarithmic, rolloff at 4.0+ |
| Line vs closed | 0.87 (topological separation) |
| Closed vs closed | 0.35 |
| Topological ratio | **2.5×** |
| Translation invariance | **720×** (V3→V4 compression) |

## Conclusions

1. V3 produces velocity tuning matching primate MT/V3 electrophysiology
2. Zero response for static stimuli — no phantom motion
3. Topological discrimination: line separated from closed shapes (2.5×)
4. V4 achieves 720× translation invariance (physics → semantics)

## Figures

- `figures/velocity_tuning_curve.png` — velocity curve + shape discrimination
