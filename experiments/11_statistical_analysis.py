"""
Experiment 11: Statistical Analysis
====================================
Confidence intervals for all key results.
Multiple random initializations (seeds) for each metric.

Addresses reviewer concern: "Most experiments do not include
formal statistical tests for comparing results."
"""
from config import *

import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import defaultdict
from scipy import stats
from src.model.substance_net import SubstanceNet
from src.data.dynamic_primitives import generate_sequence
from src.utils import generate_synthetic_cognitive_data


N_TRIALS = 5  # Number of random initializations
SEEDS = [42, 123, 456, 789, 2024]


def recognition_knn(model, train_data, test_data, N=100, test_size=1024):
    """kNN recognition test."""
    model.eval()
    def get_feat(images):
        with torch.no_grad():
            out = model(images.to(DEVICE), mode='image')
            return torch.cat([out['amplitude'], out['phase']],
                             dim=-1).mean(dim=1)
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
        if total >= test_size: break
        feats = get_feat(images)
        sims = F.cosine_similarity(
            feats.unsqueeze(1), mem_feats.unsqueeze(0), dim=2)
        topk_vals, topk_idx = sims.topk(5, dim=1)
        for i in range(feats.shape[0]):
            if total >= test_size: break
            votes = defaultdict(float)
            for j in range(5):
                votes[mem_labels[topk_idx[i, j].item()]] += \
                    topk_vals[i, j].item()
            if max(votes, key=votes.get) == labels[i].item():
                correct += 1
            total += 1
    return correct / total


def ci_95(values):
    """Compute 95% confidence interval."""
    n = len(values)
    mean = np.mean(values)
    if n < 2:
        return mean, 0.0, mean, mean
    se = stats.sem(values)
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
    return mean, se, ci[0], ci[1]


def run():
    print_header('Experiment 11: Statistical Analysis')

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(
        DATA_DIR, train=True, download=False, transform=transform)
    test_data = datasets.MNIST(
        DATA_DIR, train=False, transform=transform)

    results = {}

    # =============================================
    # 1. MNIST Backprop (5 seeds)
    # =============================================
    print(f'\n1. MNIST Backprop ({N_TRIALS} seeds):')
    mnist_accs = []
    mnist_rs = []
    for trial, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SubstanceNet(num_classes=10).to(DEVICE)
        for mod in model.modules():
            if hasattr(mod, 'set_learning'): mod.set_learning(False)
        opt = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.001)
        loader = torch.utils.data.DataLoader(
            train_data, batch_size=64, shuffle=True)
        model.train()
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images, mode='image')
            losses = model.compute_loss(out, labels)
            opt.zero_grad(); losses['total'].backward(); opt.step()
        model.eval()
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=256)
        correct, total, r_sum, n_b = 0, 0, 0.0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = model(images, mode='image')
                correct += (out['logits'].argmax(1) == labels).sum().item()
                total += labels.size(0)
                r_sum += model.get_consciousness_metrics(out)[
                    'reflexivity_score']
                n_b += 1
        acc = correct / total
        r = r_sum / n_b
        mnist_accs.append(acc)
        mnist_rs.append(r)
        print(f'  seed={seed}: acc={acc:.4f}, R={r:.4f}')

    mean, se, lo, hi = ci_95(mnist_accs)
    print(f'  Accuracy: {mean:.4f} ± {se:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])')
    mean_r, se_r, lo_r, hi_r = ci_95(mnist_rs)
    print(f'  R:        {mean_r:.4f} ± {se_r:.4f} (95% CI: [{lo_r:.4f}, {hi_r:.4f}])')
    results['mnist_backprop'] = {
        'accuracies': mnist_accs, 'Rs': mnist_rs,
        'acc_mean': mean, 'acc_se': se, 'acc_ci95': [lo, hi],
        'R_mean': mean_r, 'R_se': se_r, 'R_ci95': [lo_r, hi_r],
    }

    # =============================================
    # 2. Recognition 100-shot kNN (5 seeds)
    # =============================================
    print(f'\n2. Recognition 100-shot kNN ({N_TRIALS} seeds):')
    recog_accs = []
    for trial, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SubstanceNet(num_classes=10).to(DEVICE).eval()
        acc = recognition_knn(model, train_data, test_data, N=100)
        recog_accs.append(acc)
        print(f'  seed={seed}: {acc:.4f}')

    mean, se, lo, hi = ci_95(recog_accs)
    print(f'  Accuracy: {mean:.4f} ± {se:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])')
    results['recognition_100shot'] = {
        'accuracies': recog_accs,
        'mean': mean, 'se': se, 'ci95': [lo, hi],
    }

    # =============================================
    # 3. Cognitive Battery R (5 seeds × 10 tasks)
    # =============================================
    print(f'\n3. Cognitive Battery ({N_TRIALS} seeds):')
    tasks = ['logic', 'memory', 'categorization', 'analogy', 'spatial',
             'raven', 'numerical', 'verbal', 'emotional', 'insight']
    all_rs = []
    all_accs = []
    for trial, seed in enumerate(SEEDS):
        trial_rs = []
        trial_accs = []
        for task in tasks:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SubstanceNet(num_classes=2).to(DEVICE)
            for mod in model.modules():
                if hasattr(mod, 'set_learning'): mod.set_learning(False)
            opt = optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=0.001)
            model.train()
            for _ in range(50):
                X, y, _ = generate_synthetic_cognitive_data(
                    batch_size=32, seq_len=3, num_modules=3,
                    task_type=task)
                out = model(X.float().to(DEVICE), mode='cognitive')
                losses = model.compute_loss(out, y.to(DEVICE))
                opt.zero_grad(); losses['total'].backward(); opt.step()
            model.eval()
            with torch.no_grad():
                X_t, y_t, _ = generate_synthetic_cognitive_data(
                    batch_size=256, seq_len=3, num_modules=3,
                    task_type=task)
                out = model(X_t.float().to(DEVICE), mode='cognitive')
                acc = (out['logits'].argmax(1) == y_t.to(DEVICE)
                       ).float().mean().item()
                m = model.get_consciousness_metrics(out)
            trial_rs.append(m['reflexivity_score'])
            trial_accs.append(acc)
        all_rs.append(np.mean(trial_rs))
        all_accs.append(np.mean(trial_accs))
        print(f'  seed={seed}: acc={np.mean(trial_accs):.4f}, '
              f'R={np.mean(trial_rs):.4f} ± {np.std(trial_rs):.4f}')

    mean_r, se_r, lo_r, hi_r = ci_95(all_rs)
    mean_a, se_a, lo_a, hi_a = ci_95(all_accs)
    print(f'  Accuracy: {mean_a:.4f} ± {se_a:.4f} (95% CI: [{lo_a:.4f}, {hi_a:.4f}])')
    print(f'  R:        {mean_r:.4f} ± {se_r:.4f} (95% CI: [{lo_r:.4f}, {hi_r:.4f}])')
    results['cognitive_battery'] = {
        'Rs': all_rs, 'accs': all_accs,
        'R_mean': mean_r, 'R_se': se_r, 'R_ci95': [lo_r, hi_r],
        'acc_mean': mean_a, 'acc_se': se_a, 'acc_ci95': [lo_a, hi_a],
    }

    # =============================================
    # 4. Velocity tuning peak (5 seeds)
    # =============================================
    print(f'\n4. Velocity Tuning Peak ({N_TRIALS} seeds):')
    vel_peaks = []
    for trial, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        model = SubstanceNet(num_classes=10).to(DEVICE).eval()
        v3_out = {}
        hook = model.v3.register_forward_hook(
            lambda m, i, o: v3_out.update({'f': o}))
        frames_s, _, _ = generate_sequence(
            primitive_type=0, num_frames=6, dx=0.0, noise_std=0.0)
        with torch.no_grad():
            model(frames_s.unsqueeze(0).to(DEVICE), mode='video')
            v3_static = v3_out['f'].clone()
            frames_m, _, _ = generate_sequence(
                primitive_type=0, num_frames=6, dx=3.0, noise_std=0.0)
            model(frames_m.unsqueeze(0).to(DEVICE), mode='video')
            peak = (v3_out['f'] - v3_static).norm().item()
        hook.remove()
        vel_peaks.append(peak)
        print(f'  seed={seed}: peak={peak:.4f}')

    mean, se, lo, hi = ci_95(vel_peaks)
    print(f'  Peak: {mean:.4f} ± {se:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])')
    results['velocity_peak'] = {
        'peaks': vel_peaks,
        'mean': mean, 'se': se, 'ci95': [lo, hi],
    }

    # =============================================
    # SUMMARY PLOT
    # =============================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: MNIST accuracy
    ax = axes[0, 0]
    ax.bar(range(N_TRIALS), [a * 100 for a in mnist_accs],
           color=COLORS['primary'], alpha=0.7, edgecolor='white')
    m, _, lo, hi = ci_95(mnist_accs)
    ax.axhline(y=m * 100, color=COLORS['danger'], ls='--', lw=2)
    ax.axhspan(lo * 100, hi * 100, alpha=0.15, color=COLORS['danger'])
    ax.set_xticks(range(N_TRIALS))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel('Seed')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'MNIST Backprop: {m*100:.2f}% ± {(hi-lo)/2*100:.2f}%')
    ax.set_ylim(96, 99)

    # Panel 2: Recognition
    ax = axes[0, 1]
    ax.bar(range(N_TRIALS), [a * 100 for a in recog_accs],
           color=COLORS['success'], alpha=0.7, edgecolor='white')
    m, _, lo, hi = ci_95(recog_accs)
    ax.axhline(y=m * 100, color=COLORS['danger'], ls='--', lw=2)
    ax.axhspan(lo * 100, hi * 100, alpha=0.15, color=COLORS['danger'])
    ax.set_xticks(range(N_TRIALS))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel('Seed')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Recognition 100-shot: {m*100:.1f}% ± {(hi-lo)/2*100:.1f}%')

    # Panel 3: Cognitive R
    ax = axes[1, 0]
    ax.bar(range(N_TRIALS), all_rs,
           color=COLORS['secondary'], alpha=0.7, edgecolor='white')
    m, _, lo, hi = ci_95(all_rs)
    ax.axhline(y=m, color=COLORS['danger'], ls='--', lw=2)
    ax.axhspan(lo, hi, alpha=0.15, color=COLORS['danger'])
    ax.axhspan(0.35, 0.47, alpha=0.1, color=COLORS['success'])
    ax.set_xticks(range(N_TRIALS))
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel('Seed')
    ax.set_ylabel('R (reflexivity)')
    ax.set_title(f'Cognitive R: {m:.4f} ± {(hi-lo)/2:.4f}')

    # Panel 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Metric', 'Mean', '95% CI', 'N'],
        ['MNIST backprop',
         f'{np.mean(mnist_accs)*100:.2f}%',
         f'[{ci_95(mnist_accs)[2]*100:.2f}, {ci_95(mnist_accs)[3]*100:.2f}]',
         str(N_TRIALS)],
        ['Recognition 100-shot',
         f'{np.mean(recog_accs)*100:.1f}%',
         f'[{ci_95(recog_accs)[2]*100:.1f}, {ci_95(recog_accs)[3]*100:.1f}]',
         str(N_TRIALS)],
        ['Cognitive R',
         f'{np.mean(all_rs):.4f}',
         f'[{ci_95(all_rs)[2]:.4f}, {ci_95(all_rs)[3]:.4f}]',
         f'{N_TRIALS}×10'],
        ['Velocity peak',
         f'{np.mean(vel_peaks):.4f}',
         f'[{ci_95(vel_peaks)[2]:.4f}, {ci_95(vel_peaks)[3]:.4f}]',
         str(N_TRIALS)],
    ]
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    ax.set_title('Summary: 95% Confidence Intervals')

    fig.suptitle('Statistical Analysis (5 random initializations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure('statistical_analysis', fig)

    # === Save ===
    save_results('11_statistical_analysis', results)

    # === Summary ===
    print(f'\n{"="*60}')
    print(f'  SUMMARY: 95% Confidence Intervals')
    print(f'{"="*60}')
    print(f'  {"Metric":<25} {"Mean":>10} {"95% CI":>20} {"N":>5}')
    print(f'  {"-"*25} {"-"*10} {"-"*20} {"-"*5}')
    print(f'  {"MNIST backprop":<25} {np.mean(mnist_accs)*100:>9.2f}% '
          f'[{ci_95(mnist_accs)[2]*100:.2f}, {ci_95(mnist_accs)[3]*100:.2f}] '
          f'{N_TRIALS:>5}')
    print(f'  {"Recognition 100-shot":<25} {np.mean(recog_accs)*100:>9.1f}% '
          f'[{ci_95(recog_accs)[2]*100:.1f}, {ci_95(recog_accs)[3]*100:.1f}] '
          f'{N_TRIALS:>5}')
    print(f'  {"Cognitive R":<25} {np.mean(all_rs):>10.4f} '
          f'[{ci_95(all_rs)[2]:.4f}, {ci_95(all_rs)[3]:.4f}] '
          f'{N_TRIALS*10:>5}')
    print(f'  {"Velocity peak":<25} {np.mean(vel_peaks):>10.4f} '
          f'[{ci_95(vel_peaks)[2]:.4f}, {ci_95(vel_peaks)[3]:.4f}] '
          f'{N_TRIALS:>5}')
    print(f'{"="*60}')


if __name__ == '__main__':
    run()
