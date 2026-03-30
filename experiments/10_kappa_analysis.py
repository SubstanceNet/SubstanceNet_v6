"""
Experiment 10: κ ≈ 1 Emergence Analysis
========================================
Does SubstanceNet exhibit κ ≈ 1 plateau analogous to He-II λ-transition?

Reference: A.3_helium_lambda_kappa project
  - He-II: κ = τ × (Λ/Λᶜ) = 0.989 ± 0.007 throughout superfluid phase
  - Mechanism: ζ ≈ ν → κ ∝ t^(ζ-ν) ≈ t^0 ≈ constant

SubstanceNet analogy:
  - τ = order parameter = task performance (accuracy)
  - Λ = correlation scale = phase coherence across cliques
  - A = system capacity (amplitude)
  - κ = τ × Λ/Λᶜ × A/Aᶜ

Tests:
  1. κ stability across different cognitive tasks (analogy: temperature sweep)
  2. κ stability during training (analogy: cooling through T_λ)
  3. κ with/without consciousness module (analogy: normal vs superfluid phase)
  4. Comparison: consciousness v1 (forced R) vs v2 (emergent R)
"""
from config import *
import gc
set_seed()

import torch.optim as optim
import torch.nn.functional as F
from src.model.substance_net import SubstanceNet
from src.utils import generate_synthetic_cognitive_data


def measure_kappa_components(model, output):
    """
    Measure κ components from model output.
    
    Identification (SubstanceNet ↔ He-II):
      τ = accuracy on task (order parameter: how "ordered" is the output)
      Λ = phase coherence of consciousness (correlation scale)
      A = mean amplitude of consciousness (system capacity)
      
    κ = (A/Aᶜ) × τ × (Λ/Λᶜ)
    
    Normalization: Aᶜ and Λᶜ determined from baseline.
    """
    metrics = model.get_consciousness_metrics(output)
    
    # Extract wave state if v2
    if hasattr(model.consciousness, '_last_A'):
        A_cliques = model.consciousness._last_A      # [B, N, D]
        phi_cliques = model.consciousness._last_phi   # [B, N, D]
        
        # Λ: phase coherence across cliques
        # Mean cos(Δφ) for neighboring pairs (Hamming distance = 1/3)
        phase_diff = phi_cliques.unsqueeze(2) - phi_cliques.unsqueeze(1)
        cos_matrix = torch.cos(phase_diff)  # [B, N, N, D]
        # Take mean of off-diagonal elements
        N = phi_cliques.shape[1]
        mask = ~torch.eye(N, dtype=bool, device=phi_cliques.device)
        Lambda = cos_matrix[:, mask].mean().item()
        
        # A: mean amplitude across cliques
        A_mean = A_cliques.mean().item()
        
        # Coherence trajectory
        trajectory = model.consciousness._last_trajectory
    else:
        # v1 fallback
        Lambda = metrics.get('phase_coherence', 0.0)
        A_mean = metrics.get('mean_amplitude', 1.0)
        trajectory = []
    
    return {
        'Lambda': Lambda,
        'A': A_mean,
        'R': metrics['reflexivity_score'],
        'trajectory': trajectory,
    }


def run():
    print_header('Experiment 10: κ ≈ 1 Emergence Analysis')
    
    # =============================================
    # TEST 1: κ across cognitive tasks (trained)
    # =============================================
    print(f'\n{"="*60}')
    print(f'  TEST 1: κ across cognitive tasks')
    print(f'  Analogy: He-II temperature sweep in superfluid phase')
    print(f'{"="*60}')
    
    tasks = ['logic', 'memory', 'categorization', 'analogy', 'spatial',
             'raven', 'numerical', 'verbal', 'emotional', 'insight']
    
    results_by_version = {}
    
    for version_name, use_v2 in [('v1 (forced R)', False), 
                                   ('v2 (emergent R)', True)]:
        print(f'\n  --- {version_name} ---')
        task_results = []
        
        for task in tasks:
            set_seed()
            model = SubstanceNet(
                num_classes=2, use_consciousness_v2=use_v2).to(DEVICE)
            for mod in model.modules():
                if hasattr(mod, 'set_learning'):
                    mod.set_learning(False)
            
            optimizer = optim.Adam(
                [p for p in model.parameters() if p.requires_grad], lr=0.001)
            
            # Train
            model.train()
            for _ in range(50):
                X, y, _ = generate_synthetic_cognitive_data(
                    batch_size=32, seq_len=3, num_modules=3, task_type=task)
                out = model(X.float().to(DEVICE), mode='cognitive')
                losses = model.compute_loss(out, y.to(DEVICE))
                optimizer.zero_grad()
                losses['total'].backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                X_t, y_t, _ = generate_synthetic_cognitive_data(
                    batch_size=256, seq_len=3, num_modules=3, task_type=task)
                out = model(X_t.float().to(DEVICE), mode='cognitive')
                acc = (out['logits'].argmax(1) == y_t.to(DEVICE)
                       ).float().mean().item()
                
                kappa_data = measure_kappa_components(model, out)
            
            # τ = accuracy (order parameter)
            tau = acc
            Lambda = abs(kappa_data['Lambda'])
            A = kappa_data['A']
            
            task_results.append({
                'task': task,
                'tau': tau,
                'Lambda': Lambda,
                'A': A,
                'R': kappa_data['R'],
                'trajectory': kappa_data['trajectory'],
            })
            del model, optimizer, out
            torch.cuda.empty_cache()
            gc.collect()
        
        results_by_version[version_name] = task_results
        
        # Print results
        taus = [r['tau'] for r in task_results]
        lambdas = [r['Lambda'] for r in task_results]
        As = [r['A'] for r in task_results]
        Rs = [r['R'] for r in task_results]
        
        # Normalize: Λᶜ = max(Λ), Aᶜ = max(A)
        Lambda_c = max(lambdas) if max(lambdas) > 1e-8 else 1.0
        A_c = max(As) if max(As) > 1e-8 else 1.0
        
        kappas = [(r['A']/A_c) * r['tau'] * (abs(r['Lambda'])/Lambda_c) 
                  for r in task_results]
        
        print(f'  {"Task":<15} {"τ(acc)":>7} {"Λ":>8} {"A":>8} '
              f'{"R":>7} {"κ":>7}')
        print(f'  {"-"*15} {"-"*7} {"-"*8} {"-"*8} {"-"*7} {"-"*7}')
        for r, k in zip(task_results, kappas):
            print(f'  {r["task"]:<15} {r["tau"]:>7.4f} {r["Lambda"]:>8.4f} '
                  f'{r["A"]:>8.4f} {r["R"]:>7.4f} {k:>7.4f}')
        
        print(f'\n  κ: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}')
        print(f'  R: {np.mean(Rs):.4f} ± {np.std(Rs):.4f}')
        print(f'  Λᶜ={Lambda_c:.4f}, Aᶜ={A_c:.4f}')
    
    # =============================================
    # TEST 2: κ during training (cooling analogy)
    # =============================================
    print(f'\n{"="*60}')
    print(f'  TEST 2: κ during training')
    print(f'  Analogy: He-II cooling through T_λ')
    print(f'{"="*60}')
    
    for version_name, use_v2 in [('v1 (forced R)', False),
                                   ('v2 (emergent R)', True)]:
        print(f'\n  --- {version_name} ---')
        
        set_seed()
        model = SubstanceNet(
            num_classes=2, use_consciousness_v2=use_v2).to(DEVICE)
        for mod in model.modules():
            if hasattr(mod, 'set_learning'):
                mod.set_learning(False)
        
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.001)
        
        training_kappas = []
        training_Rs = []
        training_accs = []
        
        print(f'  {"Epoch":>6} {"Acc":>7} {"Λ":>8} {"A":>8} '
              f'{"R":>7} {"κ":>7}')
        print(f'  {"-"*6} {"-"*7} {"-"*8} {"-"*8} {"-"*7} {"-"*7}')
        
        for epoch in range(100):
            model.train()
            X, y, _ = generate_synthetic_cognitive_data(
                batch_size=32, seq_len=3, num_modules=3, task_type='logic')
            out = model(X.float().to(DEVICE), mode='cognitive')
            losses = model.compute_loss(out, y.to(DEVICE))
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
            
            # Measure every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    X_t, y_t, _ = generate_synthetic_cognitive_data(
                        batch_size=256, seq_len=3, num_modules=3,
                        task_type='logic')
                    out = model(X_t.float().to(DEVICE), mode='cognitive')
                    acc = (out['logits'].argmax(1) == y_t.to(DEVICE)
                           ).float().mean().item()
                    kd = measure_kappa_components(model, out)
                
                tau = acc
                Lambda = abs(kd['Lambda'])
                A = kd['A']
                
                # Use running normalization
                Lambda_c = max(Lambda, 1e-4)
                A_c = max(A, 1e-4)
                kappa = tau * (Lambda / Lambda_c) * (A / A_c)
                
                training_kappas.append(kappa)
                training_Rs.append(kd['R'])
                training_accs.append(acc)
                
                print(f'  {epoch+1:>6} {acc:>7.4f} {Lambda:>8.4f} '
                      f'{A:>8.4f} {kd["R"]:>7.4f} {kappa:>7.4f}')
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    # =============================================
    # TEST 3: Disabled vs enabled consciousness
    # =============================================
    print(f'\n{"="*60}')
    print(f'  TEST 3: With vs without consciousness')
    print(f'  Analogy: He-II (superfluid, κ≈1) vs He-I (normal, κ=0)')
    print(f'{"="*60}')
    
    for label, use_v2, disable_cons in [
            ('Full model (v2)', True, False),
            ('No consciousness', True, True)]:
        set_seed()
        model = SubstanceNet(
            num_classes=2, use_consciousness_v2=use_v2).to(DEVICE)
        for mod in model.modules():
            if hasattr(mod, 'set_learning'):
                mod.set_learning(False)
        
        if disable_cons:
            # Freeze consciousness — it becomes a fixed transform
            for p in model.consciousness.parameters():
                p.requires_grad = False
        
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.001)
        
        model.train()
        for _ in range(50):
            X, y, _ = generate_synthetic_cognitive_data(
                batch_size=32, seq_len=3, num_modules=3, task_type='logic')
            out = model(X.float().to(DEVICE), mode='cognitive')
            losses = model.compute_loss(out, y.to(DEVICE))
            optimizer.zero_grad()
            losses['total'].backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            X_t, y_t, _ = generate_synthetic_cognitive_data(
                batch_size=256, seq_len=3, num_modules=3, task_type='logic')
            out = model(X_t.float().to(DEVICE), mode='cognitive')
            acc = (out['logits'].argmax(1) == y_t.to(DEVICE)
                   ).float().mean().item()
            kd = measure_kappa_components(model, out)
        
        print(f'  {label:<25} acc={acc:.4f}  R={kd["R"]:.4f}  '
              f'Λ={kd["Lambda"]:.4f}  A={kd["A"]:.4f}')
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
    
    # =============================================
    # PLOTS
    # =============================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel 1: κ across tasks (both versions)
    ax = axes[0, 0]
    for vi, (version_name, task_results) in enumerate(
            results_by_version.items()):
        taus = [r['tau'] for r in task_results]
        lambdas = [abs(r['Lambda']) for r in task_results]
        As = [r['A'] for r in task_results]
        L_c = max(lambdas) if max(lambdas) > 1e-8 else 1.0
        A_c = max(As) if max(As) > 1e-8 else 1.0
        kappas = [(r['A']/A_c) * r['tau'] * (abs(r['Lambda'])/L_c)
                  for r in task_results]
        color = COLORS['primary'] if vi == 0 else COLORS['success']
        ax.plot(range(len(tasks)), kappas, 'o-', color=color,
                linewidth=2, markersize=6, label=version_name)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[:4] for t in tasks], rotation=45, fontsize=8)
    ax.set_ylabel('κ')
    ax.set_title('κ Across Cognitive Tasks')
    ax.legend(fontsize=9)
    ax.axhline(y=1.0, color=COLORS['gray'], ls='--', alpha=0.3,
               label='κ=1 target')
    
    # Panel 2: R across tasks
    ax = axes[0, 1]
    for vi, (version_name, task_results) in enumerate(
            results_by_version.items()):
        Rs = [r['R'] for r in task_results]
        color = COLORS['primary'] if vi == 0 else COLORS['success']
        ax.plot(range(len(tasks)), Rs, 'o-', color=color,
                linewidth=2, markersize=6, label=version_name)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[:4] for t in tasks], rotation=45, fontsize=8)
    ax.set_ylabel('R (reflexivity)')
    ax.set_title('Reflexivity Across Tasks')
    ax.legend(fontsize=9)
    ax.axhspan(0.35, 0.47, alpha=0.1, color=COLORS['primary'],
               label='v1 optimal')
    
    # Panel 3: Components (τ, Λ, A) for v2
    ax = axes[1, 0]
    v2_results = results_by_version.get('v2 (emergent R)', [])
    if v2_results:
        x = range(len(tasks))
        taus = [r['tau'] for r in v2_results]
        lambdas = [abs(r['Lambda']) for r in v2_results]
        As = [r['A'] for r in v2_results]
        ax.plot(x, taus, 'o-', color=COLORS['primary'], label='τ (accuracy)')
        ax.plot(x, [l * 10 for l in lambdas], 's-', color=COLORS['success'],
                label='Λ×10 (coherence)')
        ax.plot(x, As, 'D-', color=COLORS['warning'], label='A (amplitude)')
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t[:4] for t in tasks], rotation=45, fontsize=8)
    ax.set_ylabel('Component value')
    ax.set_title('κ Components (v2)')
    ax.legend(fontsize=9)
    
    # Panel 4: He-II analogy diagram
    ax = axes[1, 1]
    ax.text(0.5, 0.95, 'He-II ↔ SubstanceNet Analogy',
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=ax.transAxes)
    analogy_text = (
        'He-II (superfluid)          SubstanceNet\n'
        '─────────────────          ────────────\n'
        'T < T_λ (superfluid)  ↔  trained (learning)\n'
        'T > T_λ (normal)      ↔  untrained (random)\n'
        'τ = ρ_s/ρ (density)   ↔  τ = accuracy\n'
        'Λ = ξ (correlation)   ↔  Λ = phase coherence\n'
        'A = thermo limit      ↔  A = amplitude\n'
        'κ = τ·Λ ≈ 0.99       ↔  κ = τ·Λ·A ≈ ?\n\n'
        'Mechanism:\n'
        'He-II: ζ ≈ ν → t^(ζ-ν) ≈ const\n'
        'SubstanceNet: accuracy↑ × coherence↓\n'
        '              = compensating balance?'
    )
    ax.text(0.05, 0.80, analogy_text, ha='left', va='top',
            fontsize=9, family='monospace', transform=ax.transAxes)
    ax.axis('off')
    
    fig.suptitle('κ ≈ 1 Emergence Analysis: He-II Analogy',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure('kappa_emergence', fig)
    
    # === Save ===
    all_results = {
        'test1_tasks': {
            k: [{kk: vv for kk, vv in r.items() if kk != 'trajectory'}
                for r in v]
            for k, v in results_by_version.items()
        },
        'test2_training': {
            'kappas': training_kappas,
            'Rs': training_Rs,
            'accs': training_accs,
        },
    }
    save_results('10_kappa_analysis', all_results)
    
    # === VERDICT ===
    print(f'\n{"="*60}')
    print(f'  VERDICT: κ ≈ 1 Analysis')
    print(f'{"="*60}')
    for version_name, task_results in results_by_version.items():
        lambdas = [abs(r['Lambda']) for r in task_results]
        As = [r['A'] for r in task_results]
        L_c = max(lambdas) if max(lambdas) > 1e-8 else 1.0
        A_c = max(As) if max(As) > 1e-8 else 1.0
        kappas = [(r['A']/A_c) * r['tau'] * (abs(r['Lambda'])/L_c)
                  for r in task_results]
        Rs = [r['R'] for r in task_results]
        print(f'  {version_name}:')
        print(f'    κ = {np.mean(kappas):.4f} ± {np.std(kappas):.4f}')
        print(f'    R = {np.mean(Rs):.4f} ± {np.std(Rs):.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    run()
