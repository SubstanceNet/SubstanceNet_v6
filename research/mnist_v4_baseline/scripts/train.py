#!/usr/bin/env python3
"""
System Classification: research.mnist_v4_baseline.scripts.train
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Hypothesis:
    SubstanceNet v4 architecture reproduces v3.1.1 MNIST baseline
    (~93.74% accuracy) with optimal reflexivity R in [0.35, 0.47].

Method:
    Train SubstanceNet on MNIST with biological V1 encoding,
    quantum wave functions, and reflexive consciousness.
    Track all loss components and consciousness metrics.

Expected Output:
    - data/results.json: full training history
    - figures/: training curves, consciousness dynamics, confusion matrix
    - reports/report.md: automated analysis report

Usage:
    python scripts/train.py [--epochs 5] [--batch_size 32] [--lr 0.001]
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import SubstanceNet
from src.consciousness import TemporalConsciousnessController
from src.constants import (
    OPTIMAL_REFLEXIVITY_MIN, OPTIMAL_REFLEXIVITY_MAX,
    REFERENCE_RESULTS,
)

# Research output directories
RESEARCH_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = RESEARCH_DIR / 'data'
FIGURES_DIR = RESEARCH_DIR / 'figures'
REPORTS_DIR = RESEARCH_DIR / 'reports'


def get_dataloaders(batch_size: int = 32, data_dir: str = None):
    """Load MNIST train and test sets."""
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / 'data')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=100, shuffle=False, num_workers=2)

    return train_loader, test_loader


def evaluate(model, loader, device):
    """Evaluate model accuracy and per-class accuracy."""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output['logits'].argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            for i in range(target.size(0)):
                label = target[i].item()
                class_total[label] += 1
                if pred[i].item() == label:
                    class_correct[label] += 1

    accuracy = 100.0 * correct / total
    per_class = {str(i): (100.0 * class_correct[i] / max(class_total[i], 1))
                 for i in range(10)}

    return accuracy, per_class, np.array(all_preds), np.array(all_targets)


def train(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('=' * 70)
    print('SubstanceNet v4 — MNIST Baseline Experiment')
    print('=' * 70)
    print(f'Device: {device}')
    print(f'Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}')
    print(f'Reference: {REFERENCE_RESULTS["mnist_accuracy"]*100:.2f}% (v3.1.1)')
    print(f'Optimal reflexivity: [{OPTIMAL_REFLEXIVITY_MIN}, {OPTIMAL_REFLEXIVITY_MAX}]')
    print()

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Model
    model = SubstanceNet(
        num_classes=10,
        v1_orientations=8,
        v1_scales=3,
        v1_dim=64,
        wave_channels=128,
        consciousness_dim=32,
        abstract_dim=16,
        num_iterations=3,
    ).to(device)

    params = model.count_parameters()
    print(f'Parameters: {params["total"]:,}')
    for name, count in params.items():
        if name != 'total':
            print(f'  {name}: {count:,}')
    print()

    # Controller
    controller = TemporalConsciousnessController(mode='stream')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # === Training history ===
    history = {
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': str(device),
            'parameters': params,
            'timestamp': timestamp,
            'model_config': model.config,
        },
        'epochs': [],
        'consciousness_trace': [],
        'batch_losses': [],
    }

    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        running_cls_loss = 0.0
        running_cons_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            losses = model.compute_loss(output, target)
            losses['total'].backward()
            optimizer.step()

            # Statistics
            pred = output['logits'].argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            running_loss += losses['total'].item()
            running_cls_loss += losses['classification'].item()
            running_cons_loss += losses['consciousness'].item()
            batch_count += 1

            # Consciousness metrics every 50 batches
            if batch_idx % 50 == 0:
                metrics = model.get_consciousness_metrics(output)
                level = controller.update(metrics)

                history['consciousness_trace'].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'reflexivity': metrics['reflexivity_score'],
                    'phase_coherence': metrics['phase_coherence'],
                    'mean_amplitude': metrics['mean_amplitude'],
                    'complexity': metrics['complexity'],
                    'controller_level': level,
                    'controller_phase': controller.get_phase(),
                })

                history['batch_losses'].append({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'total': losses['total'].item(),
                    'classification': losses['classification'].item(),
                    'consciousness': losses['consciousness'].item(),
                    'zero_loss': losses['zero_loss'].item(),
                    'phase_coherence': losses['phase_coherence'].item(),
                    'topological': losses['topological'].item(),
                })

            # Progress
            if batch_idx % 200 == 0:
                acc = 100.0 * correct / total
                print(f'  Epoch {epoch} [{batch_idx:>4d}/{len(train_loader)}]  '
                      f'Loss: {losses["total"].item():.4f}  '
                      f'Acc: {acc:.1f}%  '
                      f'R: {metrics["reflexivity_score"]:.3f}')

        # Epoch summary
        train_acc = 100.0 * correct / total
        avg_loss = running_loss / batch_count
        avg_cls = running_cls_loss / batch_count
        avg_cons = running_cons_loss / batch_count

        # Test evaluation
        test_acc, per_class, test_preds, test_targets = evaluate(
            model, test_loader, device)

        epoch_time = time.time() - epoch_start

        # Controller analysis
        ctrl_analysis = controller.analyze()

        epoch_data = {
            'epoch': epoch,
            'train_accuracy': round(train_acc, 4),
            'test_accuracy': round(test_acc, 4),
            'avg_loss': round(avg_loss, 6),
            'avg_classification_loss': round(avg_cls, 6),
            'avg_consciousness_loss': round(avg_cons, 6),
            'per_class_accuracy': per_class,
            'reflexivity': round(ctrl_analysis.get('current_level', 0), 4),
            'reflexivity_phase': ctrl_analysis.get('phase', 'unknown'),
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_sec': round(epoch_time, 1),
        }
        history['epochs'].append(epoch_data)

        print(f'\n  Epoch {epoch} summary:')
        print(f'    Train: {train_acc:.2f}%  |  Test: {test_acc:.2f}%')
        print(f'    Loss: {avg_loss:.4f} (cls: {avg_cls:.4f}, cons: {avg_cons:.4f})')
        print(f'    Reflexivity: {ctrl_analysis.get("current_level", 0):.3f} '
              f'({ctrl_analysis.get("phase", "?")})')
        print(f'    Time: {epoch_time:.1f}s')
        print()

        scheduler.step()

    total_time = time.time() - total_start

    # === Final results ===
    final_acc = history['epochs'][-1]['test_accuracy']
    final_R = history['epochs'][-1]['reflexivity']
    ref_acc = REFERENCE_RESULTS['mnist_accuracy'] * 100

    history['summary'] = {
        'final_test_accuracy': final_acc,
        'final_reflexivity': final_R,
        'reference_accuracy': ref_acc,
        'accuracy_delta': round(final_acc - ref_acc, 4),
        'reflexivity_in_optimal': (
            OPTIMAL_REFLEXIVITY_MIN <= final_R <= OPTIMAL_REFLEXIVITY_MAX),
        'total_time_sec': round(total_time, 1),
        'total_parameters': params['total'],
    }

    print('=' * 70)
    print('RESULTS')
    print('=' * 70)
    print(f'  Final test accuracy: {final_acc:.2f}%')
    print(f'  Reference (v3.1.1):  {ref_acc:.2f}%')
    print(f'  Delta:               {final_acc - ref_acc:+.2f}%')
    print(f'  Reflexivity:         {final_R:.3f}  '
          f'(optimal: [{OPTIMAL_REFLEXIVITY_MIN}, {OPTIMAL_REFLEXIVITY_MAX}])')
    print(f'  R in optimal range:  {history["summary"]["reflexivity_in_optimal"]}')
    print(f'  Total time:          {total_time:.1f}s')
    print(f'  Parameters:          {params["total"]:,}')
    print()

    # === Save results ===
    results_path = DATA_DIR / f'results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    print(f'Results saved: {results_path}')

    # Also save as latest
    latest_path = DATA_DIR / 'results.json'
    with open(latest_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # === Save model checkpoint ===
    ckpt_dir = PROJECT_ROOT / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
        'history_summary': history['summary'],
    }, ckpt_dir / f'mnist_baseline_{timestamp}.pt')

    # === Generate figures ===
    generate_figures(history, timestamp)

    # === Generate report ===
    generate_report(history, timestamp)

    # === Confusion matrix data ===
    cm = np.zeros((10, 10), dtype=int)
    for pred, tgt in zip(test_preds, test_targets):
        cm[tgt][pred] += 1
    history['confusion_matrix'] = cm.tolist()
    with open(results_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # Confusion matrix figure
    generate_confusion_matrix(cm, timestamp)

    print(f'\nAll outputs in: {RESEARCH_DIR}/')
    return model, history


def generate_figures(history, timestamp):
    """Generate all training visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 11,
    })

    epochs_data = history['epochs']
    cons_trace = history['consciousness_trace']

    # === Figure 1: Training Curves (2x2) ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1a. Accuracy
    ax = axes[0, 0]
    ep = [e['epoch'] for e in epochs_data]
    ax.plot(ep, [e['train_accuracy'] for e in epochs_data],
            'b-o', linewidth=2, label='Train')
    ax.plot(ep, [e['test_accuracy'] for e in epochs_data],
            'r-s', linewidth=2, label='Test')
    ref = REFERENCE_RESULTS['mnist_accuracy'] * 100
    ax.axhline(y=ref, color='green', linestyle='--', alpha=0.7,
               label=f'v3.1.1 ref ({ref:.2f}%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy')
    ax.legend()

    # 1b. Loss components
    ax = axes[0, 1]
    if history['batch_losses']:
        steps = range(len(history['batch_losses']))
        ax.plot(steps, [b['classification'] for b in history['batch_losses']],
                label='Classification', alpha=0.8)
        ax.plot(steps, [b['consciousness'] for b in history['batch_losses']],
                label='Consciousness', alpha=0.8)
        ax.plot(steps, [b['zero_loss'] for b in history['batch_losses']],
                label='Zero loss', alpha=0.8)
    ax.set_xlabel('Training Step (x50 batches)')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Components')
    ax.legend(fontsize=9)

    # 1c. Reflexivity dynamics
    ax = axes[1, 0]
    if cons_trace:
        ax.plot([c['reflexivity'] for c in cons_trace],
                color='blue', alpha=0.7, label='Raw R')
        ax.plot([c['controller_level'] for c in cons_trace],
                color='red', linewidth=2, label='Controller R')
        ax.axhspan(OPTIMAL_REFLEXIVITY_MIN, OPTIMAL_REFLEXIVITY_MAX,
                    alpha=0.15, color='green', label='Optimal range')
    ax.set_xlabel('Training Step (x50 batches)')
    ax.set_ylabel('Reflexivity R')
    ax.set_title('Reflexivity Dynamics (Th 6.22)')
    ax.legend(fontsize=9)

    # 1d. Phase coherence & complexity
    ax = axes[1, 1]
    if cons_trace:
        ax.plot([c['phase_coherence'] for c in cons_trace],
                color='orange', label='Phase Coherence')
        ax2 = ax.twinx()
        ax2.plot([c['complexity'] for c in cons_trace],
                 color='purple', alpha=0.6, label='Complexity')
        ax2.set_ylabel('Complexity', color='purple')
        ax2.legend(loc='upper left', fontsize=9)
    ax.set_xlabel('Training Step (x50 batches)')
    ax.set_ylabel('Phase Coherence', color='orange')
    ax.set_title('Wave Function Metrics')
    ax.legend(loc='upper right', fontsize=9)

    fig.suptitle('SubstanceNet v4 — MNIST Baseline', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f'training_curves_{timestamp}.png', dpi=150)
    fig.savefig(FIGURES_DIR / 'training_curves.png', dpi=150)
    plt.close(fig)
    print(f'Figure saved: figures/training_curves.png')

    # === Figure 2: Per-class accuracy ===
    if epochs_data:
        fig, ax = plt.subplots(figsize=(10, 5))
        last_epoch = epochs_data[-1]
        classes = sorted(last_epoch['per_class_accuracy'].keys())
        accs = [last_epoch['per_class_accuracy'][c] for c in classes]
        colors = ['#2ecc71' if a >= 90 else '#e74c3c' if a < 80 else '#f39c12'
                  for a in accs]
        bars = ax.bar(classes, accs, color=colors, edgecolor='white', linewidth=0.5)
        ax.axhline(y=last_epoch['test_accuracy'], color='blue',
                    linestyle='--', label=f'Mean: {last_epoch["test_accuracy"]:.1f}%')
        ax.set_xlabel('Digit')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Per-Class Accuracy (Epoch {last_epoch["epoch"]})')
        ax.set_ylim(0, 105)
        ax.legend()
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}', ha='center', fontsize=9)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'per_class_{timestamp}.png', dpi=150)
        fig.savefig(FIGURES_DIR / 'per_class.png', dpi=150)
        plt.close(fig)
        print(f'Figure saved: figures/per_class.png')


def generate_confusion_matrix(cm, timestamp):
    """Generate confusion matrix heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color=color, fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f'confusion_matrix_{timestamp}.png', dpi=150)
    fig.savefig(FIGURES_DIR / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    print(f'Figure saved: figures/confusion_matrix.png')


def generate_report(history, timestamp):
    """Generate markdown analysis report."""
    s = history['summary']
    ep = history['epochs']
    cons = history['consciousness_trace']

    ref_acc = s['reference_accuracy']
    delta = s['accuracy_delta']
    status = 'PASS' if delta >= -2.0 else 'FAIL'
    r_status = 'PASS' if s['reflexivity_in_optimal'] else 'FAIL'

    # Consciousness analysis
    if cons:
        r_values = [c['reflexivity'] for c in cons]
        r_mean = np.mean(r_values)
        r_std = np.std(r_values)
        r_final = r_values[-1]
        coh_values = [c['phase_coherence'] for c in cons]
        coh_mean = np.mean(coh_values)
    else:
        r_mean = r_std = r_final = coh_mean = 0.0

    report = f"""# Звіт: MNIST v4 Baseline

**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Версія:** SubstanceNet v4.0.1
**Дослідник:** Oleksii Onasenko

---

## Резюме

| Метрика | Результат | Еталон | Статус |
|---------|-----------|--------|--------|
| Test Accuracy | {s['final_test_accuracy']:.2f}% | {ref_acc:.2f}% | {status} |
| Δ Accuracy | {delta:+.2f}% | — | — |
| Reflexivity R | {s['final_reflexivity']:.3f} | [0.35, 0.47] | {r_status} |
| R в оптимумі | {s['reflexivity_in_optimal']} | True | {r_status} |
| Параметри | {s['total_parameters']:,} | — | — |
| Час | {s['total_time_sec']:.0f}с | — | — |

---

## Динаміка навчання

| Епоха | Train | Test | Loss | R | Фаза |
|-------|-------|------|------|---|------|
"""
    for e in ep:
        report += (f"| {e['epoch']} | {e['train_accuracy']:.2f}% | "
                   f"{e['test_accuracy']:.2f}% | {e['avg_loss']:.4f} | "
                   f"{e['reflexivity']:.3f} | {e['reflexivity_phase']} |\n")

    report += f"""
---

## Аналіз свідомості

- **Середня рефлексивність:** {r_mean:.3f} ± {r_std:.3f}
- **Фінальна рефлексивність:** {r_final:.3f}
- **Середня когерентність:** {coh_mean:.3f}
- **Режим контролера:** stream

### Інтерпретація (Th 6.22)

Рефлексивність R = {r_final:.3f} """

    if OPTIMAL_REFLEXIVITY_MIN <= r_final <= OPTIMAL_REFLEXIVITY_MAX:
        report += "знаходиться в оптимальному діапазоні [0.35, 0.47], "
        report += "що відповідає критичному режиму κ ≈ 1."
    elif r_final < OPTIMAL_REFLEXIVITY_MIN:
        report += "нижче оптимального діапазону — недостатня глибина рефлексії."
    else:
        report += "вище оптимального діапазону — можливе насичення (Yerkes-Dodson)."

    report += f"""

---

## Точність по класах (фінальна епоха)

| Цифра | Accuracy |
|-------|----------|
"""
    if ep:
        for digit in sorted(ep[-1]['per_class_accuracy'].keys()):
            acc = ep[-1]['per_class_accuracy'][digit]
            marker = '✅' if acc >= 90 else '⚠️' if acc >= 80 else '❌'
            report += f"| {digit} | {acc:.1f}% {marker} |\n"

    report += f"""
---

## Файли

- `data/results.json` — повні результати
- `figures/training_curves.png` — криві навчання
- `figures/per_class.png` — точність по класах
- `figures/confusion_matrix.png` — матриця помилок

---

## Конфігурація моделі
```json
{json.dumps(history['config']['model_config'], indent=2)}
```

---

*Згенеровано автоматично SubstanceNet v4 research infrastructure*
"""

    report_path = REPORTS_DIR / f'report_{timestamp}.md'
    with open(report_path, 'w') as f:
        f.write(report)

    latest_path = REPORTS_DIR / 'report.md'
    with open(latest_path, 'w') as f:
        f.write(report)

    print(f'Report saved: reports/report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SubstanceNet v4 MNIST Baseline')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    model, history = train(args)
