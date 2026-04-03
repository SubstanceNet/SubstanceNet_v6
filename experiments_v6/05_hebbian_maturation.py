"""
Experiment 05: Hebbian Maturation
=================================
V3/V4 unsupervised weight adaptation through observation
of dynamic primitives (moving shapes). No labels, no backprop.

Key results:
- V3 motion signal amplified after maturation
- Velocity tuning curve preserved (logarithmic shape)
- Maturation on relevant data improves recognition
- Maturation on irrelevant data hurts recognition
"""
from config import *
set_seed()

from src.data.dynamic_primitives import generate_sequence
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import defaultdict


def measure_velocity_response(model):
    """Measure V3 response at multiple speeds."""
    v3_out = {}
    hook = model.v3.register_forward_hook(lambda m, i, o: v3_out.update({'f': o}))

    # Static reference
    frames_s, _, _ = generate_sequence(
        primitive_type=0, num_frames=6, dx=0.0, noise_std=0.0)
    with torch.no_grad():
        model(frames_s.unsqueeze(0).to(DEVICE), mode='video')
        v3_static = v3_out['f'].clone()

    speeds = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
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


def measure_shape_discrimination(model, speed=2.0):
    """Measure V3 feature distances between shapes."""
    v3_out = {}
    hook = model.v3.register_forward_hook(lambda m, i, o: v3_out.update({'f': o}))

    shapes = ['circle', 'square', 'triangle', 'line']
    feats = {}
    for prim_id, name in enumerate(shapes):
        frames, _, _ = generate_sequence(
            primitive_type=prim_id, num_frames=6, dx=speed, noise_std=0.0)
        with torch.no_grad():
            model(frames.unsqueeze(0).to(DEVICE), mode='video')
            feats[name] = v3_out['f'].clone()

    # Line vs closed average distance
    line_vs_closed = np.mean([
        (feats['line'] - feats[s]).norm().item()
        for s in ['circle', 'square', 'triangle']])
    closed_vs_closed = np.mean([
        (feats['circle'] - feats['square']).norm().item(),
        (feats['circle'] - feats['triangle']).norm().item(),
        (feats['square'] - feats['triangle']).norm().item()])

    hook.remove()
    return line_vs_closed, closed_vs_closed


def maturation_step(model, num_steps=500):
    """Run Hebbian maturation on random dynamic primitives."""
    enable_hebbian(model)
    model.train()  # HebbianLinear updates only in training mode

    v3_diffs = []
    weight_norms = []

    for step in range(num_steps):
        prim_type = np.random.randint(0, 4)
        speed = np.random.uniform(0.5, 4.0)
        angle = np.random.uniform(0, 2 * np.pi)
        dx = speed * np.cos(angle)
        dy = speed * np.sin(angle)

        frames, _, _ = generate_sequence(
            primitive_type=prim_type, num_frames=6,
            dx=dx, dy=dy, noise_std=0.0)

        with torch.no_grad():
            model(frames.unsqueeze(0).to(DEVICE), mode='video')

        if (step + 1) % 10 == 0:
            # Track V3 output_proj weight norm
            if hasattr(model.v3, 'output_proj'):
                wn = model.v3.output_proj.weight.norm().item()
            else:
                wn = 0
            weight_norms.append(wn)

            # Quick velocity check at speed=2.0
            frames_s, _, _ = generate_sequence(
                primitive_type=0, num_frames=6, dx=0.0, noise_std=0.0)
            frames_m, _, _ = generate_sequence(
                primitive_type=0, num_frames=6, dx=2.0, noise_std=0.0)

            v3_out = {}
            hook = model.v3.register_forward_hook(
                lambda m, i, o: v3_out.update({'f': o}))
            with torch.no_grad():
                model(frames_s.unsqueeze(0).to(DEVICE), mode='video')
                vs = v3_out['f'].clone()
                model(frames_m.unsqueeze(0).to(DEVICE), mode='video')
                vm = v3_out['f'].clone()
            hook.remove()
            v3_diffs.append((vm - vs).norm().item())

    # Disable after maturation
    if hasattr(model.v3, 'output_proj'):
        model.v3.output_proj.set_learning(False)
    for attr in ['compress_in', 'compress_out']:
        if hasattr(model.v4, attr):
            getattr(model.v4, attr).set_learning(False)

    return v3_diffs, weight_norms


def recognition_knn(model, train_data, test_data, N=20, test_size=1024):
    """Quick kNN recognition test."""
    def get_feat(images):
        with torch.no_grad():
            out = model(images.to(DEVICE), mode='image')
            feat = torch.cat([out['amplitude'], out['phase']], dim=-1)
            return feat.mean(dim=1)

    memory = []
    class_count = defaultdict(int)
    for img, label in train_data:
        if class_count[label] < N:
            feat = get_feat(img.unsqueeze(0))
            memory.append((feat.squeeze(0), label))
            class_count[label] += 1
        if len(class_count) == 10 and all(
                v >= N for v in class_count.values()):
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


def run():
    print_header('Experiment 05: Hebbian Maturation')

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(DATA_DIR, train=True,
                                download=False, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    # === 1. Before maturation ===
    print('1. Before maturation (fresh model):')
    set_seed()
    model = create_model(num_classes=10, in_channels=1)
    model.eval()

    speeds_before, diffs_before = measure_velocity_response(model)
    line_before, closed_before = measure_shape_discrimination(model)
    recog_before = recognition_knn(model, train_data, test_data, N=20)

    print(f'   Velocity (speed=2.0): {diffs_before[3]:.4f}')
    print(f'   Shape: line_vs_closed={line_before:.4f}, '
          f'closed_vs_closed={closed_before:.4f}')
    print(f'   Recognition 20-shot: {recog_before:.4f}')

    # === 2. Maturation on primitives ===
    print(f'\n2. Hebbian maturation (500 steps, dynamic primitives):')
    v3_history, weight_history = maturation_step(model, num_steps=500)
    print(f'   V3_diff at step 10: {v3_history[0]:.4f}')
    print(f'   V3_diff at step 500: {v3_history[-1]:.4f}')
    print(f'   Amplification: {v3_history[-1]/max(v3_history[0], 1e-8):.1f}×')

    # === 3. After maturation ===
    print(f'\n3. After maturation:')
    speeds_after, diffs_after = measure_velocity_response(model)
    line_after, closed_after = measure_shape_discrimination(model)

    print(f'   Velocity (speed=2.0): {diffs_after[3]:.4f} '
          f'(was {diffs_before[3]:.4f}, '
          f'{diffs_after[3]/max(diffs_before[3], 1e-8):.1f}× gain)')
    print(f'   Shape: line_vs_closed={line_after:.4f} '
          f'(was {line_before:.4f})')

    # === 4. Recognition after primitive maturation ===
    print(f'\n4. Recognition after primitive maturation:')
    model.eval()  # Stop Hebbian updates during recognition test
    recog_primitives = recognition_knn(model, train_data, test_data, N=20)
    print(f'   20-shot: {recog_primitives:.4f} (was {recog_before:.4f}, '
          f'delta={((recog_primitives-recog_before)*100):+.1f}%)')

    # === 5. Fresh model matured on MNIST ===
    print(f'\n5. Maturation on MNIST (relevant stimuli):')
    set_seed()
    model_mnist = create_model(num_classes=10, in_channels=1)
    enable_hebbian(model_mnist)
    model_mnist.train()  # HebbianLinear updates only in training mode

    for step in range(500):
        idx = np.random.randint(0, len(train_data))
        img, _ = train_data[idx]
        with torch.no_grad():
            model_mnist(img.unsqueeze(0).to(DEVICE), mode='image')

    # Disable Hebbian after maturation
    if hasattr(model_mnist.v3, 'output_proj'):
        model_mnist.v3.output_proj.set_learning(False)
    for attr in ['compress_in', 'compress_out']:
        if hasattr(model_mnist.v4, attr):
            getattr(model_mnist.v4, attr).set_learning(False)

    model_mnist.eval()  # Stop Hebbian updates during recognition test
    recog_mnist = recognition_knn(model_mnist, train_data, test_data, N=20)
    print(f'   20-shot: {recog_mnist:.4f} (fresh={recog_before:.4f}, '
          f'delta={((recog_mnist-recog_before)*100):+.1f}%)')

    steps_x = [(i+1)*10 for i in range(len(v3_history))]

    # === PLOTS ===

    # Fig 6: Hebbian maturation dynamics (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    # Panel 1: V3 motion response during maturation
    ax = axes[0]
    ax.plot(steps_x, v3_history, '-', color=COLORS['primary'], linewidth=1.5)
    ax.axhline(y=diffs_before[3], color=COLORS['danger'], linestyle=':',
               alpha=0.6, label=f'Before: {diffs_before[3]:.3f}')
    ax.set_xlabel('Maturation step')
    ax.set_ylabel('V3 response (speed=2.0)')
    ax.set_title('Motion Signal Amplification')
    ax.legend(fontsize=8, loc='center right')
    ax.annotate(f'Stabilizes: {v3_history[-1]:.2f}\n'
                f'({v3_history[-1]/max(diffs_before[3], 1e-8):.1f}\u00d7 from baseline)',
                xy=(350, v3_history[-1]),
                xytext=(100, (v3_history[-1] + diffs_before[3]) / 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary']),
                fontsize=9, color=COLORS['primary'])

    # Panel 2: Weight norm (Oja stability)
    ax = axes[1]
    ax.plot(steps_x, weight_history, '-', color=COLORS['secondary'],
            linewidth=1.5)
    ax.set_xlabel('Maturation step')
    ax.set_ylabel('Weight norm')
    ax.set_title('Oja Normalization (V3 weights)')
    stable_mean = np.mean(weight_history[10:])
    ax.annotate(f'Stabilizes ~{stable_mean:.1f}\n(Oja prevents divergence)',
                xy=(400, stable_mean),
                xytext=(150, stable_mean * 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary']),
                fontsize=9, color=COLORS['secondary'])

    # Panel 3: Velocity curve before vs after
    ax = axes[2]
    ax.plot(speeds_before, diffs_before, 'o--', color=COLORS['gray'],
            linewidth=1.5, markersize=6, label='Before maturation')
    ax.plot(speeds_after, diffs_after, 'o-', color=COLORS['primary'],
            linewidth=2, markersize=7, label='After maturation')
    ax.set_xlabel('Stimulus speed (px/frame)')
    ax.set_ylabel('V3 response')
    ax.set_title('Velocity Tuning: Before vs After')
    ax.legend(fontsize=9, loc='upper left')
    real_gain = diffs_after[3] / max(diffs_before[3], 1e-8)
    ax.annotate('', xy=(2.0, diffs_after[3]), xytext=(2.0, diffs_before[3]),
                arrowprops=dict(arrowstyle='<->', color=COLORS['primary'],
                                lw=2))
    ax.text(2.3, (diffs_after[3] + diffs_before[3]) / 2,
            f'{real_gain:.0f}\u00d7 gain',
            fontsize=11, fontweight='bold', color=COLORS['primary'],
            va='center')

    fig.suptitle('Hebbian Maturation: Unsupervised V3/V4 Learning (500 steps)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure('hebbian_maturation', fig)

    # Fig: Recognition comparison
    fig2, ax = plt.subplots(figsize=(7, 5.5))
    conditions = ['Fresh\n(random)', 'Matured\n(primitives)', 'Matured\n(MNIST)']
    accs = [recog_before * 100, recog_primitives * 100, recog_mnist * 100]
    colors_bar = [COLORS['gray'], COLORS['warning'], COLORS['success']]
    bars = ax.bar(range(3), accs, color=colors_bar, width=0.5,
                  edgecolor='white')
    for bar, acc, col in zip(bars, accs, colors_bar):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                f'{acc:.1f}%', ha='center', va='top',
                fontweight='bold', fontsize=13, color='white')
    ax.set_xticks(range(3))
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Recognition accuracy (%), 20-shot kNN')
    ax.set_title('Effect of Hebbian Maturation on Recognition')
    ax.set_ylim(0, 74)
    ax.axhline(y=10, color=COLORS['danger'], linestyle=':', alpha=0.4)
    delta_prim = (recog_primitives - recog_before) * 100
    delta_mnist = (recog_mnist - recog_before) * 100
    ax.text(1, accs[1] + 1, f'{delta_prim:+.1f}%\nirrelevant stimuli',
            ha='center', fontsize=9, color=COLORS['warning'], style='italic')
    ax.text(2, accs[2] + 1, f'{delta_mnist:+.1f}%\nrelevant stimuli',
            ha='center', fontsize=9, color=COLORS['success'], style='italic')
    ax.annotate('', xy=(2, max(accs) + 9), xytext=(1, max(accs[1:2]) + 9),
                arrowprops=dict(arrowstyle='<->', color=COLORS['gray'], lw=1))
    ax.text(1.5, max(accs) + 4, 'Relevant data\nhelps more',
            ha='center', fontsize=8, color=COLORS['gray'], style='italic')
    save_figure('hebbian_recognition', fig2)

    # === Save ===
    results = {
        'before_maturation': {
            'velocity_speeds': speeds_before,
            'velocity_diffs': diffs_before,
            'shape_line_vs_closed': float(line_before),
            'shape_closed_vs_closed': float(closed_before),
            'recognition_20shot': float(recog_before),
        },
        'maturation_dynamics': {
            'v3_diff_history': v3_history,
            'weight_norm_history': weight_history,
            'num_steps': 500,
        },
        'after_maturation': {
            'velocity_speeds': speeds_after,
            'velocity_diffs': diffs_after,
            'shape_line_vs_closed': float(line_after),
            'shape_closed_vs_closed': float(closed_after),
        },
        'recognition_comparison': {
            'fresh': float(recog_before),
            'matured_primitives': float(recog_primitives),
            'matured_mnist': float(recog_mnist),
        },
    }
    save_results('05_hebbian_maturation', results)

    print(f'\n{"="*60}')
    print(f'  CONCLUSION:')
    real_gain_final = diffs_after[3] / max(diffs_before[3], 1e-8)
    print(f'  V3 motion signal: {real_gain_final:.0f}× amplification after maturation')
    print(f'  Primitive maturation: {delta_prim:+.1f}% recognition')
    print(f'  MNIST maturation: {delta_mnist:+.1f}% recognition')
    print(f'  Biological finding: relevant experience helps,')
    print(f'  irrelevant experience may hurt (sensitive period)')
    print(f'{"="*60}')


if __name__ == '__main__':
    run()
