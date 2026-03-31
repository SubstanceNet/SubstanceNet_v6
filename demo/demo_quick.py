"""SubstanceNet v6 — Quick Demo (30 seconds)"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

print()
print('  SubstanceNet v6 — Quick Demo')
print('  ============================')

from src.model.substance_net import SubstanceNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SubstanceNet(num_classes=10).to(device)
total = sum(p.numel() for p in model.parameters())
print(f'  Model created: {total:,} parameters')

model.eval()
x = torch.randn(4, 1, 28, 28).to(device)
with torch.no_grad():
    out = model(x, mode='image')
    m = model.get_consciousness_metrics(out)

r = m['reflexivity_score']
ok = '✓' if 0.35 <= r <= 0.47 else '✗'
phase = 'critical' if 0.30 <= r <= 0.50 else 'non-critical'

print(f'  Forward pass: {out["logits"].shape}')
print(f'  Reflexivity R = {r:.4f} (range 0.35-0.47) {ok}')
print(f'  Phase: {phase} (κ ≈ 1)')
print(f'  Coherence: {m["phase_coherence"]:.4f}')
print()
if 0.35 <= r <= 0.47:
    print('  System status: HEALTHY (critical regime)')
else:
    print('  System status: WARNING (outside critical regime)')
print()
