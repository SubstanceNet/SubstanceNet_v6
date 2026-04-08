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
    """Extract features at specific pipeline stage.
    
    Stages (cumulative, each includes all previous):
      'v1_raw'      — V1 Gabor only (64D)
      'orient'      — + OrientationSelectivity (512D)
      'feat_proj'   — + FeatureProjection (128D) [= old 'after_wave']
      'nonlocal'    — + NonLocalInteraction (128D)
      'after_v2'    — + V2 MosaicField18 (128D)
      'after_v3'    — + V3 DynamicForm (128D)
      'after_v4'    — + V4 ObjectFeatures (128D)
    """
    # Backward compat
    if stage == 'after_wave':
        stage = 'feat_proj'

    with torch.no_grad():
        v1, _ = model.v1(images.to(DEVICE))
        if stage == 'v1_raw':
            return v1.mean(dim=1)
        o = model.orientation(v1)
        if stage == 'orient':
            return o.mean(dim=1)
        f = F.relu(model.feature_proj(o))
        if stage == 'feat_proj':
            return f.mean(dim=1)
        f = model.nonlocal_interaction(f)
        if stage == 'nonlocal':
            return f.mean(dim=1)
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

    # === TEST 5: Statistical baselines (no biological constraints) ===
    print(f'\n5. Statistical baselines (100-shot, same kNN settings):')

    from sklearn.decomposition import PCA as skPCA
    from sklearn.neighbors import KNeighborsClassifier as skKNN
    from sklearn.kernel_approximation import RBFSampler

    baseline_N = 100
    baseline_test_size = 1024

    np.random.seed(SEED)
    bl_train_idx = []
    for c in range(10):
        cls_idx = [i for i, (_, lbl) in enumerate(train_data) if lbl == c]
        bl_train_idx.extend(np.random.choice(cls_idx, baseline_N, replace=False))

    bl_X_train = np.array([train_data[i][0].numpy().flatten() for i in bl_train_idx])
    bl_y_train = np.array([train_data[i][1] for i in bl_train_idx])

    np.random.seed(SEED)
    bl_test_idx = np.random.choice(len(test_data), baseline_test_size, replace=False)
    bl_X_test = np.array([test_data[i][0].numpy().flatten() for i in bl_test_idx])
    bl_y_test = np.array([test_data[i][1] for i in bl_test_idx])

    bl_knn = skKNN(n_neighbors=5, metric='cosine', weights='distance')

    # Baseline 1: Raw pixels (784D)
    bl_knn.fit(bl_X_train, bl_y_train)
    acc_raw = bl_knn.score(bl_X_test, bl_y_test)
    print(f'   Raw Pixels (784D) + kNN:     {acc_raw*100:.1f}%')

    # Baseline 2: PCA (128D)
    pca_bl = skPCA(n_components=128, random_state=SEED)
    bl_knn.fit(pca_bl.fit_transform(bl_X_train), bl_y_train)
    acc_pca = bl_knn.score(pca_bl.transform(bl_X_test), bl_y_test)
    print(f'   PCA (128D) + kNN:            {acc_pca*100:.1f}%')

    # Baseline 3: Random Fourier Features (128D)
    rbf = RBFSampler(gamma=0.005, random_state=SEED, n_components=256)
    pca_rbf = skPCA(n_components=128, random_state=SEED)
    X_rbf_tr = pca_rbf.fit_transform(rbf.fit_transform(bl_X_train))
    X_rbf_te = pca_rbf.transform(rbf.transform(bl_X_test))
    bl_knn.fit(X_rbf_tr, bl_y_train)
    acc_rbf = bl_knn.score(X_rbf_te, bl_y_test)
    print(f'   RBF Random (128D) + kNN:     {acc_rbf*100:.1f}%')

    # Baseline 4: Fair comparison — same spatial resolution (3x3 = 9D)
    bl_X_train_9 = np.array([
        F.adaptive_avg_pool2d(
            torch.FloatTensor(train_data[i][0].numpy()).unsqueeze(0), (3, 3)
        ).flatten().numpy() for i in bl_train_idx
    ])
    bl_X_test_9 = np.array([
        F.adaptive_avg_pool2d(
            torch.FloatTensor(test_data[i][0].numpy()).unsqueeze(0), (3, 3)
        ).flatten().numpy() for i in bl_test_idx
    ])
    bl_knn.fit(bl_X_train_9, bl_y_train)
    acc_fair = bl_knn.score(bl_X_test_9, bl_y_test)
    print(f'   Fair 3x3 pool (9D) + kNN:    {acc_fair*100:.1f}%')

    print(f'   ---')
    print(f'   SubstanceNet V1->V4 (128D, 9 pos): {knn_accs[-1]*100:.1f}%')
    print(f'   Delta vs PCA: {(knn_accs[-1] - acc_pca)*100:+.1f}%'
          f' (cost of biological constraints)')
    print(f'   Delta vs Fair 3x3: {(knn_accs[-1] - acc_fair)*100:+.1f}%'
          f' (value of V1+V2 features over trivial pooling)')

    baselines = {
        'raw_pixels_784d': float(acc_raw),
        'pca_128d': float(acc_pca),
        'rbf_128d': float(acc_rbf),
        'fair_3x3_9d': float(acc_fair),
        'substancenet_128d_9pos': float(knn_accs[-1]),
        'note': 'All: 100-shot, kNN top-5 cosine weighted, test_size=1024, seed=42',
    }

    # === TEST 6: Full module ablation ===
    print(f'\n6. Full module ablation (20-shot kNN):')

    ablation_stages = [
        ('v1_raw',   'V1 Gabor only',          64,  True),
        ('orient',   'V1 + Orient',            512,  True),
        ('feat_proj','V1 + Orient + FeatProj', 128,  True),
        ('nonlocal', '+ NonLocal attention',   128,  True),
        ('after_v2', '+ V2 (3-stream)',        128,  True),
        ('after_v3', '+ V3 (Hebbian)',         128,  False),
        ('after_v4', '+ V4 (Hebbian)',         128,  False),
    ]

    ablation_accs = []
    ablation_labels = []
    ablation_dims = []
    ablation_innate = []

    for stage_key, label, dim, innate in ablation_stages:
        acc = stage_knn(model, train_data, test_data, 20, stage_key)
        ablation_accs.append(acc)
        ablation_labels.append(label)
        ablation_dims.append(dim)
        ablation_innate.append(innate)
        mark = '(innate)' if innate else '(acquired)'
        print(f'   {label:<28} {dim:>3}D  {acc*100:.1f}%  {mark}')

    # Delta analysis
    print(f'   ---')
    for i in range(1, len(ablation_accs)):
        delta = (ablation_accs[i] - ablation_accs[i-1]) * 100
        sign = '+' if delta >= 0 else ''
        print(f'   {ablation_labels[i-1]:>28} -> {ablation_labels[i]}: '
              f'{sign}{delta:.1f}%')

    ablation_results = {
        'stages': [s[0] for s in ablation_stages],
        'labels': ablation_labels,
        'dims': ablation_dims,
        'innate': ablation_innate,
        'accuracies': [float(a) for a in ablation_accs],
        'protocol': '20-shot, kNN top-5 cosine weighted, test_size=1024',
    }




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
    # Statistical baselines
    ax.axhline(y=acc_raw * 100, color='#8B0000', linestyle='-.',
               alpha=0.5, label=f'Raw pixels + kNN ({acc_raw*100:.1f}%)')
    ax.axhline(y=acc_pca * 100, color='#4B0082', linestyle='-.',
               alpha=0.5, label=f'PCA 128D + kNN ({acc_pca*100:.1f}%)')
    ax.axhline(y=acc_fair * 100, color='#006400', linestyle=':',
               alpha=0.5, label=f'Fair 3x3 + kNN ({acc_fair*100:.1f}%)')
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

    # Fig: Full module ablation
    fig_abl, ax = plt.subplots(figsize=(10, 5.5))
    x_abl = np.arange(len(ablation_accs))
    colors_abl = [COLORS['success'] if inn else COLORS['warning']
                  for inn in ablation_innate]
    bars_abl = ax.bar(x_abl, [a * 100 for a in ablation_accs],
                      color=colors_abl, width=0.6, edgecolor='white')
    for i, (bar, acc) in enumerate(zip(bars_abl, ablation_accs)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc*100:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=10)
    # Fair baseline reference
    ax.axhline(y=acc_fair * 100, color='gray', linestyle=':',
               alpha=0.5, label=f'Fair 3x3 + kNN ({acc_fair*100:.1f}%)')
    ax.axhline(y=10, color=COLORS['danger'], linestyle=':',
               alpha=0.3, label='Random (10%)')
    ax.set_xticks(x_abl)
    ax.set_xticklabels([l.replace(' + ', '\n+ ').replace('+ ', '+ ')
                         for l in ablation_labels],
                       fontsize=8, ha='center')
    ax.set_ylabel('Recognition accuracy (%), 20-shot kNN')
    ax.set_title('Module Ablation: Contribution of Each Pipeline Stage')
    ax.set_ylim(0, 78)
    from matplotlib.patches import Patch
    legend_el = [
        Patch(facecolor=COLORS['success'], label='Innate (fixed)'),
        Patch(facecolor=COLORS['warning'], label='Acquired (random weights)'),
    ]
    ax.legend(handles=legend_el + ax.get_legend_handles_labels()[0],
              loc='lower right', fontsize=8)
    save_figure('module_ablation', fig_abl)

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
        'baselines': baselines,
        'ablation': ablation_results,
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
    print(f'  Baselines: Raw={acc_raw*100:.1f}%, PCA={acc_pca*100:.1f}%, '
          f'Fair 3x3={acc_fair*100:.1f}%')
    print(f'  SubstanceNet vs PCA: {(knn_accs[-1]-acc_pca)*100:+.1f}% '
          f'(cost of biological constraints)')
    print(f'  SubstanceNet vs Fair 3x3: {(knn_accs[-1]-acc_fair)*100:+.1f}% '
          f'(value of V1+V2 features)')
    print(f'{"="*60}')


if __name__ == '__main__':
    run()
