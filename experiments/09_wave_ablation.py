"""
Experiment 09: Wave Formalism Ablation (CRITICAL)
=================================================
The central question: does ψ = A·e^(iφ) provide computational
utility beyond being a notation for two real vectors?

Ablation A (wave model):  QuantumWaveFunction + V3 phase interference
Ablation B (plain model): Two Linear layers + V3 concat+Linear

Same parameter count. Same seeds. Same tests.
If velocity tuning disappears in B — wave formalism has predictive power.
If identical — wave formalism is redundant notation.
"""
from config import *
set_seed()

import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict
from src.data.dynamic_primitives import generate_sequence
from src.model.substance_net import SubstanceNet
from src.utils import generate_synthetic_cognitive_data


# ====================================================
# Ablation: Replace wave components with plain vectors
# ====================================================

class PlainVectorFunction(nn.Module):
    """Ablation replacement for QuantumWaveFunction.
    Same architecture (two Linear layers) but no trigonometry.
    Output: two real vectors instead of A*cos(φ), A*sin(φ)."""

    def __init__(self, in_channels, out_channels, grid_size=256):
        super().__init__()
        self.half_out = out_channels // 2
        self.vec_a = nn.Linear(in_channels, self.half_out)
        self.vec_b = nn.Linear(in_channels, self.half_out)
        # Same learnable params as QuantumWaveFunction
        self.gamma_0 = nn.Parameter(torch.tensor(1e-3))
        self.epsilon = nn.Parameter(torch.tensor(1e-3))
        self.delta = nn.Parameter(torch.tensor(1e-6))

    def forward(self, oriented_features):
        a = F.relu(self.vec_a(oriented_features))  # ReLU instead of softplus+cos
        b = self.vec_b(oriented_features)           # plain linear instead of A*sin(φ)
        # Return dummy complex (zeros) + two real vectors as "amplitude" and "phase"
        psi_complex = torch.complex(a, b)
        return psi_complex, a, b

    def zero_loss(self, amplitude, phase):
        # Simple L2 regularization instead of topological zero loss
        if amplitude.shape[1] <= 1:
            return torch.tensor(0.0, device=amplitude.device)
        grad = amplitude[:, 1:] - amplitude[:, :-1]
        return self.gamma_0 * torch.mean(grad ** 2)


class PlainV3(nn.Module):
    """Ablation replacement for V3 temporal interference.
    Same dims but uses concat+Linear instead of interference formula."""

    def __init__(self, dim, num_streams=3, phase_dim=64):
        super().__init__()
        self.dim = dim
        self.stream_size = dim // num_streams
        self.phase_dim = phase_dim

        # Replace interference with learned linear combination
        self.motion_proj = nn.Linear(self.stream_size, phase_dim)
        self.form_proj = nn.Linear(self.stream_size, phase_dim)
        self.combine = nn.Linear(dim + phase_dim, dim)
        self.output_norm = nn.LayerNorm(dim)
        self.residual_gate = nn.Parameter(torch.tensor(0.5))

        # Static mode (same as original)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.stream_size * 2, self.stream_size),
                nn.Sigmoid(),
            ) for _ in range(num_streams)
        ])
        self.static_output = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU())

    def forward_temporal(self, v2_sequence, amplitude_sequence=None,
                         phase_sequence=None):
        B, T, seq_len, dim = v2_sequence.shape
        last_v2 = v2_sequence[:, -1]

        thick_start = 0
        pale_start = self.stream_size * 2
        thick_t = v2_sequence[..., thick_start:thick_start + self.stream_size]
        form = last_v2[..., pale_start:pale_start + self.stream_size]

        # Motion = temporal diff (same as original)
        thick_diff = thick_t[:, 1:] - thick_t[:, :-1]
        motion = thick_diff.mean(dim=1)

        # ABLATION: concat + Linear instead of interference formula
        form_feat = self.form_proj(form)
        motion_feat = self.motion_proj(motion)
        combined_feat = F.relu(form_feat + motion_feat)

        combined = torch.cat([last_v2, combined_feat], dim=-1)
        projected = self.combine(combined)
        integrated = F.relu(self.output_norm(projected))

        g = torch.sigmoid(self.residual_gate)
        return g * integrated + (1 - g) * last_v2

    def forward_static(self, x):
        streams = x.split(self.stream_size, dim=-1)
        if len(streams) > 3:
            streams = list(streams[:2]) + [torch.cat(list(streams[2:]), dim=-1)]
        streams = list(streams)
        gated = []
        for i in range(min(len(streams), 3)):
            j = (i + 1) % len(streams)
            s_i = streams[i][..., :self.stream_size]
            s_j = streams[j][..., :self.stream_size]
            gate = self.gates[i](torch.cat([s_i, s_j], dim=-1))
            gated.append(gate * s_i)
        combined = torch.cat(gated, dim=-1)
        if combined.shape[-1] < self.dim:
            pad = torch.zeros(*combined.shape[:-1],
                              self.dim - combined.shape[-1],
                              device=x.device, dtype=x.dtype)
            combined = torch.cat([combined, pad], dim=-1)
        integrated = self.static_output(combined)
        g = torch.sigmoid(self.residual_gate)
        return g * integrated + (1 - g) * x

    def forward(self, x, amplitude_sequence=None, phase_sequence=None):
        if x.dim() == 4 and phase_sequence is not None:
            return self.forward_temporal(x, amplitude_sequence, phase_sequence)
        elif x.dim() == 4:
            return self.forward_static(x[:, -1])
        return self.forward_static(x)


def create_ablation_model():
    """Create SubstanceNet with wave components replaced by plain vectors."""
    set_seed()
    model = SubstanceNet(num_classes=10, in_channels=1).to(DEVICE)

    # Replace QuantumWaveFunction with PlainVectorFunction
    plain_wave = PlainVectorFunction(
        in_channels=model.wave.in_channels,
        out_channels=model.wave.out_channels,
    ).to(DEVICE)
    model.wave = plain_wave

    # Replace V3 with PlainV3
    plain_v3 = PlainV3(
        dim=model.v3.dim,
        phase_dim=model.v3.phase_dim,
    ).to(DEVICE)
    model.v3 = plain_v3

    # Disable Hebbian (for fair comparison)
    for mod in model.modules():
        if hasattr(mod, 'set_learning'):
            mod.set_learning(False)

    return model


# ====================================================
# Test functions
# ====================================================

def test_velocity(model, label):
    """Measure V3 velocity tuning curve."""
    model.eval()
    v3_out = {}
    hook = model.v3.register_forward_hook(
        lambda m, i, o: v3_out.update({'f': o}))

    frames_s, _, _ = generate_sequence(
        primitive_type=0, num_frames=6, dx=0.0, noise_std=0.0)
    with torch.no_grad():
        model(frames_s.unsqueeze(0).to(DEVICE), mode='video')
        v3_static = v3_out['f'].clone()

    speeds = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    diffs = []
    for speed in speeds:
        frames, _, _ = generate_sequence(
            primitive_type=0, num_frames=6, dx=speed, noise_std=0.0)
        with torch.no_grad():
            model(frames.unsqueeze(0).to(DEVICE), mode='video')
            diff = (v3_out['f'] - v3_static).norm().item()
        diffs.append(diff)

    hook.remove()
    return speeds, diffs


def test_shape_discrimination(model):
    """Measure V3 shape discrimination."""
    model.eval()
    v3_out = {}
    hook = model.v3.register_forward_hook(
        lambda m, i, o: v3_out.update({'f': o}))

    shapes = ['circle', 'square', 'triangle', 'line']
    feats = {}
    for pid, name in enumerate(shapes):
        frames, _, _ = generate_sequence(
            primitive_type=pid, num_frames=6, dx=2.0, noise_std=0.0)
        with torch.no_grad():
            model(frames.unsqueeze(0).to(DEVICE), mode='video')
            feats[name] = v3_out['f'].clone()

    line_vs_closed = np.mean([
        (feats['line'] - feats[s]).norm().item()
        for s in ['circle', 'square', 'triangle']])
    closed_vs_closed = np.mean([
        (feats['circle'] - feats['square']).norm().item(),
        (feats['circle'] - feats['triangle']).norm().item(),
        (feats['square'] - feats['triangle']).norm().item()])

    hook.remove()
    return line_vs_closed, closed_vs_closed


def test_recognition(model, train_data, test_data, N=20, test_size=1024):
    """kNN recognition test."""
    model.eval()

    def get_feat(images):
        with torch.no_grad():
            out = model(images.to(DEVICE), mode='image')
            feat = torch.cat([out['amplitude'], out['phase']], dim=-1)
            return feat.mean(dim=1)

    memory = []
    cc = defaultdict(int)
    for img, label in train_data:
        if cc[label] < N:
            memory.append((get_feat(img.unsqueeze(0)).squeeze(0), label))
            cc[label] += 1
        if len(cc) == 10 and all(v >= N for v in cc.values()):
            break

    mem_feats = torch.stack([m[0] for m in memory])
    mem_labels = [m[1] for m in memory]

    loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False)
    correct, total = 0, 0
    for images, labels in loader:
        if total >= test_size:
            break
        feats = get_feat(images)
        sims = F.cosine_similarity(
            feats.unsqueeze(1), mem_feats.unsqueeze(0), dim=2)
        topk_vals, topk_idx = sims.topk(5, dim=1)
        for i in range(feats.shape[0]):
            if total >= test_size:
                break
            votes = defaultdict(float)
            for j in range(5):
                votes[mem_labels[topk_idx[i, j].item()]] += \
                    topk_vals[i, j].item()
            if max(votes, key=votes.get) == labels[i].item():
                correct += 1
            total += 1
    return correct / total


def test_mnist_backprop(model, train_data, test_data, epochs=1):
    """Quick MNIST training test."""
    model.train()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.001)
    loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True)

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        out = model(images, mode='image')
        losses = model.compute_loss(out, labels)
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

    model.eval()
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=256, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images, mode='image')
            correct += (out['logits'].argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def test_cognitive(model, task='logic', epochs=50):
    """Quick cognitive task test."""
    model.train()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.001)

    for _ in range(epochs):
        X, y, _ = generate_synthetic_cognitive_data(
            batch_size=32, seq_len=3, num_modules=3, task_type=task)
        out = model(X.float().to(DEVICE), mode='cognitive')
        losses = model.compute_loss(out, y.to(DEVICE))
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_t, y_t, _ = generate_synthetic_cognitive_data(
            batch_size=256, seq_len=3, num_modules=3, task_type=task)
        out = model(X_t.float().to(DEVICE), mode='cognitive')
        acc = (out['logits'].argmax(1) == y_t.to(DEVICE)).float().mean().item()
        m = model.get_consciousness_metrics(out)
    return acc, m['reflexivity_score']


# ====================================================
# Main experiment
# ====================================================

def run():
    print_header('Experiment 09: Wave Formalism Ablation')

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(DATA_DIR, train=True,
                                download=False, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    # === Create both models ===
    print('Creating models:')
    set_seed()
    wave_model = create_model(num_classes=10, in_channels=1)
    wave_params = sum(p.numel() for p in wave_model.parameters())
    print(f'  Wave model:  {wave_params:,} parameters')

    set_seed()
    plain_model = create_ablation_model()
    plain_params = sum(p.numel() for p in plain_model.parameters())
    print(f'  Plain model: {plain_params:,} parameters')
    print(f'  Difference:  {abs(wave_params - plain_params):,}')

    results = {'wave': {}, 'plain': {}}

    # === 1. Velocity tuning ===
    print(f'\n{"="*55}')
    print(f'  1. VELOCITY TUNING CURVE')
    print(f'{"="*55}')

    sp_w, df_w = test_velocity(wave_model, 'Wave')
    sp_p, df_p = test_velocity(plain_model, 'Plain')

    print(f'  {"Speed":>6}  {"Wave":>8}  {"Plain":>8}  {"Ratio":>7}')
    print(f'  {"-"*6}  {"-"*8}  {"-"*8}  {"-"*7}')
    for i, speed in enumerate(sp_w):
        ratio = df_w[i] / max(df_p[i], 1e-8) if df_p[i] > 0.001 else 'N/A'
        r_str = f'{ratio:.1f}×' if isinstance(ratio, float) else ratio
        print(f'  {speed:>6.1f}  {df_w[i]:>8.4f}  {df_p[i]:>8.4f}  {r_str:>7}')

    results['wave']['velocity'] = {'speeds': sp_w, 'diffs': df_w}
    results['plain']['velocity'] = {'speeds': sp_p, 'diffs': df_p}

    # === 2. Shape discrimination ===
    print(f'\n{"="*55}')
    print(f'  2. SHAPE DISCRIMINATION (speed=2.0)')
    print(f'{"="*55}')

    lvc_w, cvc_w = test_shape_discrimination(wave_model)
    lvc_p, cvc_p = test_shape_discrimination(plain_model)

    print(f'  {"Metric":<25}  {"Wave":>8}  {"Plain":>8}')
    print(f'  {"-"*25}  {"-"*8}  {"-"*8}')
    print(f'  {"Line vs closed":<25}  {lvc_w:>8.4f}  {lvc_p:>8.4f}')
    print(f'  {"Closed vs closed":<25}  {cvc_w:>8.4f}  {cvc_p:>8.4f}')
    print(f'  {"Ratio (topological)":<25}  '
          f'{lvc_w/max(cvc_w,1e-8):>8.1f}×  {lvc_p/max(cvc_p,1e-8):>8.1f}×')

    results['wave']['shape'] = {'line_vs_closed': float(lvc_w),
                                 'closed_vs_closed': float(cvc_w)}
    results['plain']['shape'] = {'line_vs_closed': float(lvc_p),
                                  'closed_vs_closed': float(cvc_p)}

    # === 3. Recognition ===
    print(f'\n{"="*55}')
    print(f'  3. RECOGNITION (20-shot kNN)')
    print(f'{"="*55}')

    set_seed()
    recog_w = test_recognition(wave_model, train_data, test_data, N=20)
    set_seed()
    recog_p = test_recognition(plain_model, train_data, test_data, N=20)

    print(f'  Wave model:  {recog_w:.4f} ({recog_w*100:.1f}%)')
    print(f'  Plain model: {recog_p:.4f} ({recog_p*100:.1f}%)')
    print(f'  Delta: {(recog_w - recog_p)*100:+.1f}%')

    results['wave']['recognition_20shot'] = float(recog_w)
    results['plain']['recognition_20shot'] = float(recog_p)

    # === 4. MNIST backprop ===
    print(f'\n{"="*55}')
    print(f'  4. MNIST BACKPROP (1 epoch)')
    print(f'{"="*55}')

    set_seed()
    wave_bp = create_model(num_classes=10, in_channels=1)
    mnist_w = test_mnist_backprop(wave_bp, train_data, test_data)

    set_seed()
    plain_bp = create_ablation_model()
    mnist_p = test_mnist_backprop(plain_bp, train_data, test_data)

    print(f'  Wave model:  {mnist_w:.4f} ({mnist_w*100:.1f}%)')
    print(f'  Plain model: {mnist_p:.4f} ({mnist_p*100:.1f}%)')
    print(f'  Delta: {(mnist_w - mnist_p)*100:+.1f}%')

    results['wave']['mnist_backprop'] = float(mnist_w)
    results['plain']['mnist_backprop'] = float(mnist_p)

    # === 5. Cognitive task (R stability) ===
    print(f'\n{"="*55}')
    print(f'  5. COGNITIVE TASK (R stability)')
    print(f'{"="*55}')

    set_seed()
    cog_acc_w, cog_r_w = test_cognitive(
        create_model(num_classes=2, in_channels=1))
    set_seed()
    cog_acc_p, cog_r_p = test_cognitive(create_ablation_model())

    print(f'  {"Metric":<20}  {"Wave":>8}  {"Plain":>8}')
    print(f'  {"-"*20}  {"-"*8}  {"-"*8}')
    print(f'  {"Accuracy":<20}  {cog_acc_w:>8.4f}  {cog_acc_p:>8.4f}')
    print(f'  {"Reflexivity R":<20}  {cog_r_w:>8.4f}  {cog_r_p:>8.4f}')

    results['wave']['cognitive'] = {'accuracy': float(cog_acc_w),
                                     'R': float(cog_r_w)}
    results['plain']['cognitive'] = {'accuracy': float(cog_acc_p),
                                      'R': float(cog_r_p)}

    # === PLOTS ===

    # Fig 9: Ablation comparison (2 panels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Velocity tuning curves
    ax1.plot(sp_w, df_w, 'o-', color=COLORS['primary'], linewidth=2,
             markersize=7, label='Wave (ψ = A·e^(iφ))')
    ax1.plot(sp_p, df_p, 's--', color=COLORS['danger'], linewidth=2,
             markersize=7, label='Plain (two Linear layers)')
    ax1.set_xlabel('Stimulus speed (px/frame)')
    ax1.set_ylabel('V3 response (L2 norm diff)')
    ax1.set_title('Velocity Tuning: Wave vs Plain')
    ax1.legend(fontsize=9)
    ax1.axhline(y=0, color=COLORS['gray'], linestyle='-', alpha=0.2)

    # Right: Summary bar chart
    metrics = ['Velocity\n(speed=3)', 'Shape\n(line vs closed)',
               'Recognition\n(20-shot)', 'MNIST\n(backprop)',
               'Cognitive\n(accuracy)']
    wave_vals = [
        df_w[sp_w.index(3.0)],
        lvc_w,
        recog_w,
        mnist_w,
        cog_acc_w,
    ]
    plain_vals = [
        df_p[sp_p.index(3.0)],
        lvc_p,
        recog_p,
        mnist_p,
        cog_acc_p,
    ]

    # Normalize each to wave=1.0
    ratios_wave = [1.0] * 5
    ratios_plain = [p / max(w, 1e-8) for w, p in zip(wave_vals, plain_vals)]

    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width/2, ratios_wave, width, color=COLORS['primary'],
            label='Wave', edgecolor='white')
    ax2.bar(x + width/2, ratios_plain, width, color=COLORS['danger'],
            label='Plain', edgecolor='white')
    ax2.axhline(y=1.0, color=COLORS['gray'], linestyle='--', alpha=0.4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylabel('Relative to Wave model (=1.0)')
    ax2.set_title('Wave vs Plain: All Metrics')
    ax2.legend(fontsize=9)

    # Annotate significant differences
    for i, (w, p) in enumerate(zip(wave_vals, plain_vals)):
        if w > 0:
            pct = (p - w) / w * 100
            color = COLORS['success'] if pct > 5 else \
                    COLORS['danger'] if pct < -5 else COLORS['gray']
            ax2.text(i + width/2, ratios_plain[i] + 0.03,
                     f'{pct:+.0f}%', ha='center', fontsize=8,
                     color=color, fontweight='bold')

    plt.tight_layout()
    save_figure('wave_ablation', fig)

    # === Save ===
    results['parameter_counts'] = {
        'wave': wave_params,
        'plain': plain_params,
    }
    save_results('09_wave_ablation', results)

    # === Verdict ===
    peak_w = max(df_w)
    peak_p = max(df_p)
    vel_ratio = peak_w / max(peak_p, 1e-8)

    print(f'\n{"="*55}')
    print(f'  VERDICT: WAVE FORMALISM ABLATION')
    print(f'{"="*55}')
    print(f'  Velocity tuning peak: Wave={peak_w:.4f} vs Plain={peak_p:.4f}'
          f' ({vel_ratio:.1f}×)')
    print(f'  Shape discrimination: Wave={lvc_w:.4f} vs Plain={lvc_p:.4f}')
    print(f'  Recognition:          Wave={recog_w:.1%} vs Plain={recog_p:.1%}')
    print(f'  MNIST backprop:       Wave={mnist_w:.1%} vs Plain={mnist_p:.1%}')
    print(f'  Cognitive R:          Wave={cog_r_w:.4f} vs Plain={cog_r_p:.4f}')
    print()
    if vel_ratio > 2.0:
        print(f'  → Wave formalism produces {vel_ratio:.0f}× stronger velocity')
        print(f'    tuning than plain vectors. FORMALISM HAS PREDICTIVE POWER.')
    elif vel_ratio > 1.2:
        print(f'  → Moderate advantage ({vel_ratio:.1f}×). Suggestive but not')
        print(f'    conclusive. More tests needed.')
    else:
        print(f'  → No significant difference ({vel_ratio:.1f}×).')
        print(f'    Wave formalism may be redundant notation.')
    print(f'{"="*55}')


if __name__ == '__main__':
    run()
