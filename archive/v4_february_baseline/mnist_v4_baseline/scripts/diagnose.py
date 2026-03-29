#!/usr/bin/env python3
"""
System Classification: research.mnist_v4_baseline.scripts.diagnose
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Diagnostic: Is the architecture actually working as a whole?

Tests:
1. Gradient flow — do gradients reach V1 and wave modules?
2. Feature variance — are wave/consciousness outputs informative?
3. Ablation — what happens if we randomize intermediate features?
4. Representation analysis — do features change with different digits?
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src import SubstanceNet

RESEARCH_DIR = Path(__file__).resolve().parent.parent


def get_test_batch(n=100):
    """Get a batch of MNIST test images."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = torchvision.datasets.MNIST(
        root=str(PROJECT_ROOT / 'data'), train=False,
        download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    return next(iter(loader))


def test_gradient_flow(model, data, target):
    """Test 1: Do gradients flow through ALL modules?"""
    print('=' * 60)
    print('TEST 1: GRADIENT FLOW')
    print('=' * 60)

    model.train()
    model.zero_grad()

    output = model(data)
    loss = F.cross_entropy(output['logits'], target)
    loss.backward()

    results = {}
    for name, module in [
        ('v1.gabor_bank', model.v1.gabor_bank),
        ('v1.simple_cells', model.v1.simple_cells),
        ('v1.complex_cells', model.v1.complex_cells),
        ('v1.hypercolumns', model.v1.hypercolumns),
        ('orientation', model.orientation),
        ('wave.amplitude_net', model.wave.amplitude_fc),
        ('wave.phase_net', model.wave.phase_fc),
        ('nonlocal.attn', model.nonlocal_interaction.attn),
        ('abstraction', model.abstraction),
        ('consciousness.project', model.consciousness.projection),
        ('consciousness.evolve', model.consciousness.F_transform),
        ('classifier', model.classifier),
    ]:
        grad_norms = []
        for p in module.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.abs().mean().item())

        if grad_norms:
            mean_grad = np.mean(grad_norms)
            max_grad = np.max(grad_norms)
            status = '✅ ACTIVE' if mean_grad > 1e-7 else '❌ DEAD'
        else:
            mean_grad = max_grad = 0.0
            status = '⚠️  NO PARAMS'

        results[name] = {
            'mean_grad': mean_grad,
            'max_grad': max_grad,
            'status': status,
        }
        print(f'  {name:30s}  mean={mean_grad:.2e}  max={max_grad:.2e}  {status}')

    return results


def test_feature_variance(model, data, target):
    """Test 2: Are intermediate representations informative?"""
    print()
    print('=' * 60)
    print('TEST 2: FEATURE VARIANCE (are features informative?)')
    print('=' * 60)

    model.eval()
    with torch.no_grad():
        output = model(data)

    results = {}
    for name, tensor in [
        ('logits', output['logits']),
        ('abstract', output['abstract']),
        ('amplitude', output['amplitude']),
        ('phase', output['phase']),
        ('amplitude_c', output['amplitude_c']),
        ('phase_c', output['phase_c']),
    ]:
        var = tensor.var().item()
        mean = tensor.abs().mean().item()
        std = tensor.std().item()

        # Check if all samples produce the same output (collapsed)
        per_sample_mean = tensor.view(tensor.shape[0], -1).mean(dim=1)
        sample_var = per_sample_mean.var().item()

        collapsed = sample_var < 1e-6
        status = '❌ COLLAPSED' if collapsed else '✅ DIVERSE'

        results[name] = {
            'variance': var,
            'mean_abs': mean,
            'std': std,
            'sample_variance': sample_var,
            'collapsed': collapsed,
            'status': status,
        }
        print(f'  {name:15s}  var={var:.4e}  mean={mean:.4f}  '
              f'sample_var={sample_var:.4e}  {status}')

    return results


def test_ablation_shuffle(model, data, target):
    """Test 3: Does shuffling intermediate features hurt accuracy?
    
    If architecture works as a whole, destroying intermediate
    representations should significantly drop accuracy.
    If classifier ignores them, accuracy won't change much.
    """
    print()
    print('=' * 60)
    print('TEST 3: ABLATION — SHUFFLE INTERMEDIATE FEATURES')
    print('=' * 60)

    model.eval()
    results = {}

    # Normal accuracy
    with torch.no_grad():
        output = model(data)
        normal_acc = (output['logits'].argmax(1) == target).float().mean().item() * 100

    print(f'  Normal accuracy: {normal_acc:.1f}%')
    results['normal'] = normal_acc

    # Hook-based ablation: shuffle features at different stages
    hooks = []

    def make_shuffle_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                first = output[0]
                idx = torch.randperm(first.shape[0])
                return (first[idx],) + output[1:]
            else:
                idx = torch.randperm(output.shape[0])
                return output[idx]
        return hook

    def make_zero_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            else:
                return torch.zeros_like(output)
        return hook

    def make_random_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.randn_like(output[0]),) + output[1:]
            else:
                return torch.randn_like(output)
        return hook

    # Test each module
    ablation_targets = [
        ('v1', model.v1),
        ('orientation', model.orientation),
        ('wave', model.wave),
        ('nonlocal', model.nonlocal_interaction),
        ('abstraction', model.abstraction),
    ]

    for module_name, module in ablation_targets:
        # Random noise ablation
        h = module.register_forward_hook(make_random_hook(module_name))
        with torch.no_grad():
            try:
                output = model(data)
                abl_acc = (output['logits'].argmax(1) == target).float().mean().item() * 100
            except Exception as e:
                abl_acc = -1
                print(f'  {module_name:20s}  ERROR: {e}')
        h.remove()

        drop = normal_acc - abl_acc
        connected = drop > 5.0
        status = '✅ CONNECTED' if connected else '❌ DISCONNECTED'

        results[module_name] = {
            'accuracy_with_noise': abl_acc,
            'accuracy_drop': drop,
            'connected': connected,
        }
        print(f'  Noise in {module_name:15s}: acc={abl_acc:5.1f}%  '
              f'drop={drop:+5.1f}%  {status}')

    return results


def test_digit_representations(model, data, target):
    """Test 4: Do different digits produce different representations?"""
    print()
    print('=' * 60)
    print('TEST 4: DIGIT-SPECIFIC REPRESENTATIONS')
    print('=' * 60)

    model.eval()
    with torch.no_grad():
        output = model(data)

    results = {}

    for feat_name in ['abstract', 'amplitude', 'phase']:
        feat = output[feat_name].view(data.shape[0], -1)

        # Compute mean representation per digit
        digit_means = []
        for d in range(10):
            mask = target == d
            if mask.sum() > 0:
                digit_means.append(feat[mask].mean(dim=0))

        if len(digit_means) >= 2:
            digit_means = torch.stack(digit_means)
            # Inter-class distance
            dists = torch.cdist(digit_means, digit_means)
            mean_inter = dists[dists > 0].mean().item()

            # Intra-class variance
            intra_vars = []
            for d in range(10):
                mask = target == d
                if mask.sum() > 1:
                    intra_vars.append(feat[mask].var(dim=0).mean().item())
            mean_intra = np.mean(intra_vars) if intra_vars else 0

            # Separation ratio (higher = better separation)
            ratio = mean_inter / (np.sqrt(mean_intra) + 1e-8)

            results[feat_name] = {
                'inter_class_dist': mean_inter,
                'intra_class_var': mean_intra,
                'separation_ratio': ratio,
            }
            print(f'  {feat_name:12s}  inter_dist={mean_inter:.4f}  '
                  f'intra_var={mean_intra:.4e}  separation={ratio:.2f}')

    return results


def run_all():
    """Run complete diagnostics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('SubstanceNet v4 — Architecture Diagnostics')
    print(f'Device: {device}')
    print()

    # Load trained model if available, otherwise use untrained
    model = SubstanceNet(num_classes=10).to(device)

    ckpt_dir = PROJECT_ROOT / 'checkpoints'
    ckpts = sorted(ckpt_dir.glob('mnist_baseline_*.pt'))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f'Loaded checkpoint: {ckpts[-1].name}')
    else:
        print('WARNING: No checkpoint found, using UNTRAINED model')
    print()

    data, target = get_test_batch(200)
    data, target = data.to(device), target.to(device)

    # Run all tests
    all_results = {}
    all_results['gradient_flow'] = test_gradient_flow(model, data, target)
    all_results['feature_variance'] = test_feature_variance(model, data, target)
    all_results['ablation'] = test_ablation_shuffle(model, data, target)
    all_results['representations'] = test_digit_representations(model, data, target)

    # Summary
    print()
    print('=' * 60)
    print('DIAGNOSTIC SUMMARY')
    print('=' * 60)

    # Gradient check
    dead_modules = [k for k, v in all_results['gradient_flow'].items()
                    if 'DEAD' in v['status']]
    if dead_modules:
        print(f'  ❌ Dead gradient modules: {dead_modules}')
    else:
        print(f'  ✅ All modules receive gradients')

    # Collapse check
    collapsed = [k for k, v in all_results['feature_variance'].items()
                 if v.get('collapsed', False)]
    if collapsed:
        print(f'  ❌ Collapsed features: {collapsed}')
    else:
        print(f'  ✅ All features are diverse across samples')

    # Connection check
    disconnected = [k for k, v in all_results['ablation'].items()
                    if isinstance(v, dict) and not v.get('connected', True)]
    if disconnected:
        print(f'  ❌ Disconnected modules: {disconnected}')
        print(f'     (noise in these modules does NOT affect accuracy)')
    else:
        print(f'  ✅ All modules connected to classification')

    # Overall verdict
    print()
    issues = len(dead_modules) + len(collapsed) + len(disconnected)
    if issues == 0:
        print('  VERDICT: Architecture works as integrated whole ✅')
    else:
        print(f'  VERDICT: {issues} issue(s) found — architecture partially disconnected ⚠️')

    # Save
    results_path = RESEARCH_DIR / 'data' / f'diagnostics_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  Saved: {results_path}')


if __name__ == '__main__':
    run_all()
