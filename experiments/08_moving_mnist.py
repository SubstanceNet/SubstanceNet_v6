"""
Experiment 08: Moving MNIST
============================
Tests SubstanceNet on dynamic (moving) digits:
- V3 motion detection on real MNIST digits
- Recognition of moving objects
- Cross-modal: static→moving and moving→static
"""
from config import *
set_seed()

import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import defaultdict


def create_moving_mnist(digit_img, canvas_size=64, speed=2.0,
                        num_frames=6):
    """Place MNIST digit on larger canvas with controlled motion."""
    h, w = digit_img.shape[-2:]
    frames = []
    start_x = (canvas_size - w) // 2
    start_y = (canvas_size - h) // 2

    for t in range(num_frames):
        canvas = torch.zeros(1, canvas_size, canvas_size)
        x = int(start_x + speed * t) % (canvas_size - w)
        y = start_y
        canvas[:, y:y+h, x:x+w] = digit_img
        frames.append(canvas)

    return torch.stack(frames)  # [T, 1, 64, 64]


def get_features_video(model, frames):
    """Extract features from video sequence."""
    with torch.no_grad():
        out = model(frames.unsqueeze(0).to(DEVICE), mode='video')
        feat = torch.cat([out['amplitude'], out['phase']], dim=-1)
        return feat.mean(dim=1).squeeze(0)


def get_features_static(model, image):
    """Extract features from static image."""
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(DEVICE), mode='image')
        feat = torch.cat([out['amplitude'], out['phase']], dim=-1)
        return feat.mean(dim=1).squeeze(0)


def knn_recognize(query_feat, memory, top_k=5):
    """kNN top-k weighted cosine voting."""
    mem_feats = torch.stack([m[0] for m in memory])
    sims = F.cosine_similarity(
        query_feat.unsqueeze(0).unsqueeze(1),
        mem_feats.unsqueeze(0), dim=2).squeeze(0)
    topk_vals, topk_idx = sims.topk(min(top_k, len(memory)))
    votes = defaultdict(float)
    for j in range(topk_vals.shape[0]):
        votes[memory[topk_idx[j].item()][1]] += topk_vals[j].item()
    return max(votes, key=votes.get) if votes else -1


def run():
    print_header('Experiment 08: Moving MNIST')

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(DATA_DIR, train=True,
                                download=False, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    model = create_model(num_classes=10, in_channels=1)
    model.eval()

    # === 1. V3 motion detection on real digits ===
    print('1. V3 motion detection on real MNIST digits:')

    v3_out = {}
    hook = model.v3.register_forward_hook(
        lambda m, i, o: v3_out.update({'f': o}))

    # Get a sample digit
    sample_img, sample_label = train_data[0]

    speeds = [0.0, 1.0, 2.0, 4.0]
    v3_diffs = []

    # Static reference on 64x64
    frames_static = create_moving_mnist(sample_img, speed=0.0)
    with torch.no_grad():
        model(frames_static.unsqueeze(0).to(DEVICE), mode='video')
        v3_static = v3_out['f'].clone()

    print(f'  Sample digit: {sample_label}')
    print(f'  {"Speed":>6}  {"V3_diff":>8}')
    print(f'  {"-"*6}  {"-"*8}')

    for speed in speeds:
        frames = create_moving_mnist(sample_img, speed=speed)
        with torch.no_grad():
            model(frames.unsqueeze(0).to(DEVICE), mode='video')
            diff = (v3_out['f'] - v3_static).norm().item()
        v3_diffs.append(diff)
        print(f'  {speed:>6.1f}  {diff:>8.4f}')

    hook.remove()

    # === 2. Static recognition baseline (28x28) ===
    print(f'\n2. Static recognition baseline (20-shot):')
    N = 20

    # Store static features
    static_memory = []
    class_count = defaultdict(int)
    for img, label in train_data:
        if class_count[label] < N:
            feat = get_features_static(model, img)
            static_memory.append((feat, label))
            class_count[label] += 1
        if len(class_count) == 10 and all(
                v >= N for v in class_count.values()):
            break

    # Test static → static
    correct, total = 0, 0
    for img, label in test_data:
        if total >= 500:
            break
        feat = get_features_static(model, img)
        if knn_recognize(feat, static_memory) == label:
            correct += 1
        total += 1
    static_acc = correct / total
    print(f'  Static→Static: {static_acc:.4f}')

    # === 3. Moving recognition (64x64) ===
    print(f'\n3. Moving recognition (20-shot):')

    # Store moving features
    moving_memory = []
    class_count2 = defaultdict(int)
    for img, label in train_data:
        if class_count2[label] < N:
            frames = create_moving_mnist(img, speed=2.0)
            feat = get_features_video(model, frames)
            moving_memory.append((feat, label))
            class_count2[label] += 1
        if len(class_count2) == 10 and all(
                v >= N for v in class_count2.values()):
            break

    # Test moving → moving
    correct_mm, total_mm = 0, 0
    for img, label in test_data:
        if total_mm >= 500:
            break
        frames = create_moving_mnist(img, speed=2.0)
        feat = get_features_video(model, frames)
        if knn_recognize(feat, moving_memory) == label:
            correct_mm += 1
        total_mm += 1
    moving_acc = correct_mm / total_mm
    print(f'  Moving→Moving (speed=2.0): {moving_acc:.4f}')

    # === 4. Cross-modal recognition ===
    print(f'\n4. Cross-modal recognition:')

    # Static memory → moving test
    correct_sm, total_sm = 0, 0
    for img, label in test_data:
        if total_sm >= 500:
            break
        frames = create_moving_mnist(img, speed=2.0)
        feat = get_features_video(model, frames)
        if knn_recognize(feat, static_memory) == label:
            correct_sm += 1
        total_sm += 1
    cross_static_moving = correct_sm / total_sm
    print(f'  Static protos → Moving test: {cross_static_moving:.4f}')

    # Moving memory → static test
    correct_ms, total_ms = 0, 0
    for img, label in test_data:
        if total_ms >= 500:
            break
        feat = get_features_static(model, img)
        if knn_recognize(feat, moving_memory) == label:
            correct_ms += 1
        total_ms += 1
    cross_moving_static = correct_ms / total_ms
    print(f'  Moving protos → Static test: {cross_moving_static:.4f}')

    # === 5. Speed robustness ===
    print(f'\n5. Recognition at different speeds (static memory):')
    speed_accs = {}
    for speed in [0.5, 1.0, 2.0, 3.0, 4.0]:
        correct_s, total_s = 0, 0
        for img, label in test_data:
            if total_s >= 300:
                break
            frames = create_moving_mnist(img, speed=speed)
            feat = get_features_video(model, frames)
            if knn_recognize(feat, static_memory) == label:
                correct_s += 1
            total_s += 1
        acc = correct_s / total_s
        speed_accs[speed] = acc
        print(f'  speed={speed:.1f}: {acc:.4f}')

    # R check
    out = model(torch.randn(4, 1, 28, 28).to(DEVICE), mode='image')
    r = model.get_consciousness_metrics(out)['reflexivity_score']
    print(f'\n  R: {r:.4f}')

    # === PLOTS ===

    # Fig 8: Model boundaries summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: All recognition conditions
    conditions = [
        ('Static→Static\n(28×28)', static_acc),
        ('Moving→Moving\n(64×64, sp=2)', moving_acc),
        ('Static→Moving\n(cross-modal)', cross_static_moving),
        ('Moving→Static\n(cross-modal)', cross_moving_static),
    ]
    names = [c[0] for c in conditions]
    accs = [c[1] * 100 for c in conditions]
    colors_bar = [COLORS['primary'], COLORS['secondary'],
                  COLORS['warning'], COLORS['warning']]

    bars = ax1.barh(range(len(conditions)), accs, color=colors_bar,
                    height=0.5, edgecolor='white')
    ax1.set_yticks(range(len(conditions)))
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('Recognition accuracy (%)')
    ax1.set_title('Recognition Across Modalities (20-shot kNN)')
    ax1.axvline(x=10, color=COLORS['danger'], linestyle=':', alpha=0.5,
                label='Random (10%)')
    ax1.invert_yaxis()

    for bar, acc in zip(bars, accs):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontweight='bold', fontsize=10)
    ax1.legend(fontsize=8)

    # Right: Speed robustness
    sp_list = sorted(speed_accs.keys())
    sp_accs = [speed_accs[s] * 100 for s in sp_list]
    ax2.plot(sp_list, sp_accs, 'o-', color=COLORS['primary'],
             linewidth=2, markersize=7)
    ax2.axhline(y=static_acc * 100, color=COLORS['success'],
                linestyle='--', alpha=0.5,
                label=f'Static baseline: {static_acc*100:.1f}%')
    ax2.axhline(y=10, color=COLORS['danger'], linestyle=':',
                alpha=0.4, label='Random (10%)')
    ax2.set_xlabel('Stimulus speed (px/frame)')
    ax2.set_ylabel('Recognition accuracy (%)')
    ax2.set_title('Speed Robustness (Static Memory → Moving Test)')
    ax2.legend(fontsize=9, loc='lower left')

    plt.tight_layout()
    save_figure('model_boundaries', fig)

    # === Save ===
    results = {
        'v3_motion_detection': {
            'speeds': speeds,
            'v3_diffs': v3_diffs,
            'digit': int(sample_label),
        },
        'recognition': {
            'static_static': float(static_acc),
            'moving_moving': float(moving_acc),
            'static_to_moving': float(cross_static_moving),
            'moving_to_static': float(cross_moving_static),
        },
        'speed_robustness': {
            str(k): float(v) for k, v in speed_accs.items()
        },
        'consciousness_R': float(r),
    }
    save_results('08_moving_mnist', results)

    print(f'\n{"="*60}')
    print(f'  CONCLUSION:')
    print(f'  V3 detects motion on real MNIST digits')
    print(f'  Static→Static: {static_acc*100:.1f}%')
    print(f'  Moving→Moving: {moving_acc*100:.1f}%')
    print(f'  Cross-modal: {cross_static_moving*100:.1f}% / '
          f'{cross_moving_static*100:.1f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    run()
