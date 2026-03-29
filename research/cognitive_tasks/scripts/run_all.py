"""
SubstanceNet v4 — Cognitive Task Battery
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
from src.utils import generate_synthetic_cognitive_data
_gen = generate_synthetic_cognitive_data
import torch.optim as optim
import json
from datetime import datetime
from src.model.substance_net import SubstanceNet

TASKS = ["logic", "memory", "categorization", "analogy",
         "spatial", "raven", "numerical", "verbal",
         "emotional", "insight"]

from src.consciousness.controller import TemporalConsciousnessController

def train_task(task_type, epochs=100, batch_size=32, lr=0.001, device="cuda"):
    model = SubstanceNet(num_classes=2, abstract_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    controller = TemporalConsciousnessController(mode='stream')
    history = []

    for epoch in range(epochs):
        model.train()
        X, y, abs_y = _gen(batch_size=batch_size, seq_len=3,
                           num_modules=3, task_type=task_type)
        X, y = X.float().to(device), y.to(device)

        output = model(X, mode="cognitive")
        losses = model.compute_loss(output, y)
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Update controller with current metrics
        metrics = model.get_consciousness_metrics(output)
        controller.update(metrics)

        if (epoch + 1) % 20 == 0:
            acc = (output['logits'].argmax(1) == y).float().mean().item()
            r_val = losses.get('current_r', metrics['reflexivity_score'])
            history.append({
                'epoch': epoch + 1,
                'acc': acc,
                'R': r_val,
                'R_controlled': controller.current_level,
                'coherence': metrics['phase_coherence'],
                'phase': controller.get_phase(),
            })

    # Final test
    model.eval()
    with torch.no_grad():
        X_t, y_t, _ = _gen(batch_size=256, seq_len=3,
                            num_modules=3, task_type=task_type)
        X_t, y_t = X_t.float().to(device), y_t.to(device)
        out = model(X_t, mode="cognitive")
        test_acc = (out['logits'].argmax(1) == y_t).float().mean().item()
        m = model.get_consciousness_metrics(out)
        abs_var = out['abstract'].var(dim=0).mean().item()

    ctrl_analysis = controller.analyze()

    return {
        'task': task_type,
        'test_accuracy': test_acc,
        'reflexivity': m['reflexivity_score'],
        'R_controlled': ctrl_analysis.get('current_level', 0),
        'R_phase': ctrl_analysis.get('phase', 'unknown'),
        'phase_coherence': m['phase_coherence'],
        'mean_amplitude': m['mean_amplitude'],
        'complexity': m['complexity'],
        'abstract_variance': abs_var,
        'history': history,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("SubstanceNet v4 — Cognitive Task Battery")
    print(f"Device: {device}")
    print("=" * 70)

    results = {}
    for task in TASKS:
        print(f"\n--- {task.upper()} ---")
        r = train_task(task, epochs=100, batch_size=32, device=device)
        results[task] = r
        print(f"  Accuracy:     {r['test_accuracy']:.4f}")
        print(f"  Reflexivity:  {r['reflexivity']:.4f}")
        print(f"  Coherence:    {r['phase_coherence']:.4f}")
        print(f"  Amplitude:    {r['mean_amplitude']:.4f}")
        print(f"  Complexity:   {r['complexity']:.1f}")
        print(f"  Abstract var: {r['abstract_variance']:.2e}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Task':<16} {'Acc':>7} {'R':>7} {'R_ctrl':>7} {'Phase':<14} {'Coh':>7} {'AbsVar':>10}")
    print("-" * 80)
    acc_s, r_s, rc_s, c_s = 0, 0, 0, 0
    for task in TASKS:
        r = results[task]
        acc_s += r['test_accuracy']
        r_s += r['reflexivity']
        rc_s += r.get('R_controlled', 0)
        c_s += r['phase_coherence']
        print(f"{task:<16} {r['test_accuracy']:>7.4f} {r['reflexivity']:>7.4f} "
              f"{r.get('R_controlled', 0):>7.4f} {r.get('R_phase', '?'):<14} "
              f"{r['phase_coherence']:>7.4f} {r['abstract_variance']:>10.2e}")
    n = len(TASKS)
    print("-" * 80)
    print(f"{'MEAN':<16} {acc_s/n:>7.4f} {r_s/n:>7.4f} {rc_s/n:>7.4f}")
    print(f"\n--- v3.1.1 Reference ---")
    print(f"  Mean accuracy:     0.9934")
    print(f"  Mean reflexivity:  0.9563")
    print(f"  Mean coherence:    0.9980")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"research/cognitive_tasks/results_{ts}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")

if __name__ == "__main__":
    main()
