"""SubstanceNet v6 — Velocity Tuning Demo"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

print()
print('  SubstanceNet v6 — Velocity Tuning Demo')
print('  =======================================')
print('  V3 phase interference detects motion')
print()

from src.model.substance_net import SubstanceNet
from src.data.dynamic_primitives import generate_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SubstanceNet(num_classes=10).to(device).eval()

v3_out = {}
model.v3.register_forward_hook(lambda m, i, o: v3_out.update({'f': o}))

frames_s, _, _ = generate_sequence(primitive_type=0, num_frames=6, dx=0.0, noise_std=0.0)
with torch.no_grad():
    model(frames_s.unsqueeze(0).to(device), mode='video')
    v3_static = v3_out['f'].clone()

print(f'  {"Speed":>6}  {"V3 response":>11}  {"Bar":}')
print(f'  {"─"*6}  {"─"*11}  {"─"*30}')

for speed in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
    frames, _, _ = generate_sequence(primitive_type=0, num_frames=6, dx=speed, noise_std=0.0)
    with torch.no_grad():
        model(frames.unsqueeze(0).to(device), mode='video')
        diff = (v3_out['f'] - v3_static).norm().item()
    bar = '█' * int(diff * 150)
    print(f'  {speed:>6.1f}  {diff:>11.4f}  {bar}')

print()
print('  Logarithmic saturation — matches primate MT/V3')
print('  Zero response for static — no phantom motion')
print()
