"""
Experiment 03: Recognition Paradigm
====================================
EXACT methodology: top-k=5 weighted cosine voting on amplitude+phase (128-dim).
This is kNN in feature space, not prototype matching.

Key insight: kNN dramatically outperforms prototypes at higher N
because it preserves within-class variability (different handwriting styles).
"""
from config import *

import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import defaultdict


def get_features(model, images):
    """Extract amplitude+phase features (128-dim) via V1->V4."""
    with torch.no_grad():
        out = model(images.to(DEVICE), mode='image')
        feat = torch.cat([out['amplitude'], out['phase']], dim=-1)
        return feat.mean(dim=1)


def knn_recognition(model, train_data, test_data, N,
                    top_k=5, test_size=1024):
    """
    Recognition via kNN top-k weighted cosine voting.
    Stores N individual episodes per class, retrieves top-k nearest,
    votes weighted by cosine similarity.
    """
    # Store N episodes per class
    memory = []
    class_count = defaultdict(int)
    for img, label in train_data:
        if class_count[label] < N:
            feat = get_features(model, img.unsqueeze(0))
            memory.append((feat.squeeze(0), label))
            class_count[label] += 1
        if len(class_count) == 10 and all(
                v >= N for v in class_count.values()):
            break

    mem_feats = torch.stack([m[0] for m in memory])
    mem_labels = [m[1] for m in memory]

    # Test via top-k voting
    loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False)
    correct, total = 0, 0
    for images, labels in loader:
        if total >= test_size:
            break
        feats = get_features(model, images)
        sims = F.cosine_similarity(
            feats.unsqueeze(1), mem_feats.unsqueeze(0), dim=2)
        topk_vals, topk_idx = sims.topk(
            min(top_k, len(memory)), dim=1)
        for i in range(feats.shape[0]):
            if total >= test_size:
                break
            votes = defaultdict(float)
            for j in range(topk_vals.shape[1]):
                votes[mem_labels[topk_idx[i, j].item()]] += \
                    topk_vals[i, j].item()
            if max(votes, key=votes.get) == labels[i].item():
                correct += 1
            total += 1

    return correct / total, len(memory)


def prototype_recognition(model, train_data, test_data, N,
                          test_size=1024):
    """Prototype matching: mean per class, cosine similarity."""
    class_feats = defaultdict(list)
    for img, label in train_data:
        if len(class_feats[label]) < N:
            feat = get_features(model, img.unsqueeze(0))
            class_feats[label].append(feat.squeeze(0))
        if len(class_feats) == 10 and all(
                len(v) >= N for v in class_feats.values()):
            break

    protos = torch.stack([
        torch.stack(class_feats[i]).mean(0) for i in range(10)])

    loader = torch.utils.data.DataLoader(
        test_data, batch_size=128, shuffle=False)
    correct, total = 0, 0
    for images, labels in loader:
        if total >= test_size:
            break
        feats = get_features(model, images)
        sims = F.cosine_similarity(
            feats.unsqueeze(1), protos.unsqueeze(0), dim=2)
        correct += (sims.argmax(1).cpu() == labels).sum().item()
        total += feats.shape[0]
    return correct / total


def get_features_at_stage(model, images, stage):
    """Extract features at specific V1->V4 stage."""
    with torch.no_grad():
        v1, _ = model.v1(images.to(DEVICE))
        o = model.orientation(v1)
        f = F.relu(model.feature_proj(o))
        half = f.shape[-1] // 2
        a, p = f[..., :half], f[..., half:]
        if stage == 'after_wave':
            return f.mean(dim=1)
        f = model.nonlocal_interaction(f)
        v2 = model.v2(f)
        if stage == 'after_v2':
            return v2.mean(dim=1)
        v3 = model.v3(v2)
        if stage == 'after_v3':
            return v3.mean(dim=1)
        v4 = model.v4(v3)
        return v4.mean(dim=1)


def stage_knn(model, train_data, test_data, N, stage,
              top_k=5, test_size=1024):
    """kNN recognition at specific pipeline stage."""
    feat_fn = lambda imgs: get_features_at_stage(model, imgs, stage)

    memory = []
    class_count = defaultdict(int)
    for img, label in train_data:
        if class_count[label] < N:
            feat = feat_fn(img.unsqueeze(0))
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
        feats = feat_fn(images)
        sims = F.cosine_similarity(
            feats.unsqueeze(1), mem_feats.unsqueeze(0), dim=2)
        topk_vals, topk_idx = sims.topk(
            min(top_k, len(memory)), dim=1)
        for i in range(feats.shape[0]):
            if total >= test_size:
                break
            votes = defaultdict(float)
            for j in range(topk_vals.shape[1]):
                votes[mem_labels[topk_idx[i, j].item()]] += \
                    topk_vals[i, j].item()
            if max(votes, key=votes.get) == labels[i].item():
                correct += 1
            total += 1
    return correct / total


def run():
    print_header('Experiment 03: Recognition Paradigm (kNN voting)')

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(DATA_DIR, train=True,
                                download=True, transform=transform)
    test_data = datasets.MNIST(DATA_DIR, train=False, transform=transform)

    # === 1. N-shot kNN scaling ===
    print('1. N-shot recognition (kNN top-5 weighted voting):')
    set_seed()
    model = create_model(num_classes=10, in_channels=1)
    model.eval()

    n_shots = [1, 5, 10, 20, 50, 100]
    knn_accs = []
    proto_accs = []

    for N in n_shots:
        knn_acc, mem_size = knn_recognition(
            model, train_data, test_data, N, test_size=1024)
        proto_acc = prototype_recognition(
            model, train_data, test_data, N, test_size=1024)
        knn_accs.append(knn_acc)
        proto_accs.append(proto_acc)
        print(f'   {N:>3}-shot: kNN={knn_acc:.4f}  proto={proto_acc:.4f}'
              f'  delta={((knn_acc-proto_acc)*100):+.1f}%  [{mem_size} episodes]')

    # === 2. Reproducibility across seeds ===
    print(f'\n2. Reproducibility (100-shot, 5 random inits):')
    trial_accs = []
    for trial in range(5):
        m = create_model(num_classes=10, in_channels=1)
        m.eval()
        acc, _ = knn_recognition(m, train_data, test_data, 100,
                                  test_size=1024)
        trial_accs.append(acc)
        print(f'   trial {trial}: {acc:.4f}')
    print(f'   mean: {np.mean(trial_accs):.4f} ± {np.std(trial_accs):.4f}')

    # === 3. Consolidation: kNN vs prototypes ===
    print(f'\n3. Consolidation (kNN episodes vs prototypes):')
    set_seed()
    model = create_model(num_classes=10, in_channels=1)
    model.eval()
    N_cons = 20
    knn_acc_20, mem = knn_recognition(
        model, train_data, test_data, N_cons, test_size=1024)
    proto_acc_20 = prototype_recognition(
        model, train_data, test_data, N_cons, test_size=1024)
    print(f'   kNN ({mem} episodes):    {knn_acc_20:.4f}')
    print(f'   Prototypes (10 means):  {proto_acc_20:.4f}')
    print(f'   Memory ratio: {mem}:10 = {mem//10}×')
    print(f'   Accuracy trade-off: {(proto_acc_20-knn_acc_20)*100:+.1f}% for'
          f' {mem//10}× compression')

    # === 4. Innate vs acquired (kNN at each stage) ===
    print(f'\n4. Feature quality by stage (20-shot kNN):')
    set_seed()
    model = create_model(num_classes=10, in_channels=1)
    model.eval()

    stages = ['after_wave', 'after_v2', 'after_v3', 'after_v4']
    stage_labels = ['V1 (Gabor)', 'V2 (FFT+diff)',
                    'V3 (interference)', 'V4 (attention)']
    stage_innate = [True, True, False, False]
    stage_accs = []

    for stage, label in zip(stages, stage_labels):
        acc = stage_knn(model, train_data, test_data, 20, stage,
                        test_size=1024)
        stage_accs.append(acc)
        mark = '(innate)' if stage in ['after_wave', 'after_v2'] \
            else '(acquired)'
        print(f'   {label:<22} {acc:.4f}  {mark}')

    # R check
    out = model(torch.randn(4, 1, 28, 28).to(DEVICE), mode='image')
    r = model.get_consciousness_metrics(out)['reflexivity_score']
    print(f'\n   R: {r:.4f}')

    # === PLOTS ===

    # Fig 4: Recognition Scaling (kNN vs prototypes)
    fig1, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(n_shots, [a * 100 for a in knn_accs], 'o-',
            color=COLORS['primary'], linewidth=2, markersize=7,
            label='kNN top-5 voting (seed=42)', zorder=3)
    ax.plot(n_shots, [a * 100 for a in proto_accs], 's--',
            color=COLORS['secondary'], linewidth=1.5, markersize=6,
            alpha=0.7, label='Prototype matching (seed=42)')
    ax.errorbar(100, np.mean(trial_accs) * 100,
                yerr=np.std(trial_accs) * 100,
                fmt='D', color=COLORS['success'], markersize=10,
                capsize=5, capthick=2, linewidth=2, zorder=4,
                label=f'Mean 5 random inits: '
                      f'{np.mean(trial_accs)*100:.1f}% '
                      f'\u00b1 {np.std(trial_accs)*100:.1f}%')
    ax.axhline(y=10, color=COLORS['danger'], linestyle=':',
               alpha=0.5, label='Random baseline (10%)')
    ax.axhline(y=97.43, color=COLORS['gray'], linestyle='--',
               alpha=0.4, label='Backprop reference (97.4%)')
    ax.set_xscale('log')
    ax.set_xlabel('Examples per class (N)')
    ax.set_ylabel('Recognition accuracy (%)')
    ax.set_title('Recognition Without Backpropagation: kNN vs Prototypes')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xticks(n_shots)
    ax.set_xticklabels([str(n) for n in n_shots])
    ax.set_ylim(0, 100)
    ax.annotate('kNN preserves\nwithin-class variability',
                xy=(50, knn_accs[4] * 100), xytext=(8, 70),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'],
                                lw=1.5),
                fontsize=9, color=COLORS['primary'])
    ax.annotate('Prototypes destroy\nindividual differences',
                xy=(50, proto_accs[4] * 100), xytext=(8, 30),
                arrowprops=dict(arrowstyle='->', color=COLORS['secondary'],
                                lw=1.5),
                fontsize=9, color=COLORS['secondary'], alpha=0.8)
    save_figure('recognition_scaling', fig1)

    # Fig 5: Innate vs Acquired
    fig2, ax = plt.subplots(figsize=(8, 5.5))
    x_st = np.arange(len(stages))
    colors_bars = [COLORS['success'] if inn else COLORS['warning']
                   for inn in stage_innate]
    bars = ax.bar(x_st, [a * 100 for a in stage_accs], color=colors_bars,
                  width=0.55, edgecolor='white', linewidth=0.5)
    for i, (bar, acc) in enumerate(zip(bars, stage_accs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                f'{acc*100:.1f}%', ha='center', va='top',
                fontweight='bold', fontsize=13, color='white')
    ax.axhline(y=10, color=COLORS['danger'], linestyle=':', alpha=0.4)
    ax.set_xticks(x_st)
    ax.set_xticklabels(['V1\n(Gabor)', 'V2\n(FFT+diff)',
                         'V3\n(interference)', 'V4\n(attention)'])
    ax.set_ylabel('Recognition accuracy (%), 20-shot kNN')
    ax.set_title('Feature Quality by Processing Stage')
    ax.set_ylim(0, 78)
    # V1→V2 improvement annotation (between bars, above both)
    v2_top = stage_accs[1] * 100
    ax.annotate('', xy=(1, v2_top + 1), xytext=(0, v2_top + 1),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'],
                                lw=2))
    ax.text(0.5, v2_top + 2.5,
            f'+{(stage_accs[1]-stage_accs[0])*100:.0f}%  FFT+diff, no training',
            ha='center', fontsize=9, fontweight='bold',
            color=COLORS['success'], style='italic')
    ax.text(3, stage_accs[3] * 100 + 7,
            'V4 random weights:\nattention without training hurts',
            fontsize=8, color=COLORS['gray'], ha='center', style='italic')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['success'], label='Innate (fixed, no training)'),
        Patch(facecolor=COLORS['warning'], label='Acquired (random weights)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    save_figure('innate_vs_acquired', fig2)

    # Fig 7: Consolidation trade-off
    fig3, ax = plt.subplots(figsize=(6.5, 5.5))
    bars = ax.bar([0, 1], [knn_acc_20 * 100, proto_acc_20 * 100],
                  color=[COLORS['primary'], COLORS['secondary']],
                  width=0.45, edgecolor='white')
    for bar, acc in zip(bars, [knn_acc_20, proto_acc_20]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc*100:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=13, color=bar.get_facecolor())
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'kNN episodes\n({mem} memories)',
                         f'Prototypes\n(10 memories)'])
    ax.set_ylabel('Recognition accuracy (%), 20-shot')
    ax.set_title('Memory Trade-off: Episodes vs Prototypes')
    ax.set_ylim(0, 70)
    delta = (proto_acc_20 - knn_acc_20) * 100
    ax.text(0.5, 0.97,
            f'{mem//10}\u00d7 memory compression  |  {delta:+.1f}% accuracy trade-off',
            transform=ax.transAxes, ha='center', fontsize=10,
            style='italic', color=COLORS['gray'])
    save_figure('consolidation', fig3)

    # === Save ===
    results = {
        'n_shot_scaling': {
            'n_shots': n_shots,
            'knn_accuracies': [float(a) for a in knn_accs],
            'prototype_accuracies': [float(a) for a in proto_accs],
            'method': 'kNN top-5 weighted cosine voting (128-dim amp+phase)',
        },
        'reproducibility': {
            'trial_accuracies': [float(a) for a in trial_accs],
            'mean': float(np.mean(trial_accs)),
            'std': float(np.std(trial_accs)),
        },
        'consolidation': {
            'knn_acc': float(knn_acc_20),
            'prototype_acc': float(proto_acc_20),
            'knn_episodes': mem,
            'prototype_memories': 10,
        },
        'stage_features': {
            stage: float(acc) for stage, acc in zip(stages, stage_accs)
        },
        'consciousness_R': float(r),
    }
    save_results('03_recognition_paradigm', results)

    print(f'\n{"="*60}')
    print(f'  CONCLUSION:')
    best_knn = max(knn_accs); best_n = n_shots[knn_accs.index(best_knn)]
    print(f'  kNN recognition: {best_n}-shot = {best_knn*100:.1f}%'
          f' ({best_knn*10:.1f}× random)')
    print(f'  Reproducibility: {np.mean(trial_accs)*100:.1f}%'
          f' ± {np.std(trial_accs)*100:.1f}% (5 trials)')
    print(f'  V1+V2 innate: {stage_accs[1]*100:.1f}% without training')
    print(f'  kNN vs proto at 20-shot: {(knn_acc_20-proto_acc_20)*100:+.1f}%')
    print(f'{"="*60}')


if __name__ == '__main__':
    run()
