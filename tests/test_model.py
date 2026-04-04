"""
System Classification: tests.test_model
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
License: Apache-2.0

Tests for SubstanceNet model.
Updated: 2026-03-18 (V2/V3/V4/Hippocampus integration, HebbianLinear, R-targeting)
"""
import torch
import pytest


@pytest.fixture
def model():
    from src.model import SubstanceNet
    m = SubstanceNet(num_classes=10)
    # Disable Hebbian learning to avoid inplace ops during grad tests
    if hasattr(m, 'v3') and hasattr(m.v3, 'output_proj'):
        m.v3.output_proj.set_learning(False)
    if hasattr(m, 'v4'):
        for attr in ['compress_in', 'compress_out']:
            if hasattr(m.v4, attr):
                getattr(m.v4, attr).set_learning(False)
    return m


@pytest.fixture
def mnist_batch():
    return torch.randn(4, 1, 28, 28)


def test_forward_output_keys(model, mnist_batch):
    """Forward pass returns all expected keys."""
    output = model(mnist_batch)
    expected = {'logits', 'abstract', 'psi_c', 'amplitude_c',
                'phase_c', 'amplitude', 'phase', 'v1_activations'}
    assert expected == set(output.keys())


def test_forward_logits_shape(model, mnist_batch):
    """Logits have correct shape [B, num_classes]."""
    output = model(mnist_batch)
    assert output['logits'].shape == (4, 10)


def test_forward_consciousness_complex(model, mnist_batch):
    """Consciousness output is complex-valued."""
    output = model(mnist_batch)
    assert output['psi_c'].is_complex()


def test_compute_loss(model, mnist_batch):
    """Loss computation produces all components."""
    output = model(mnist_batch)
    target = torch.randint(0, 10, (4,))
    losses = model.compute_loss(output, target)

    # Required loss components (current architecture)
    required = {'total', 'classification', 'abstract', 'consciousness',
                'phase_coherence', 'topological', 'zero_loss',
                'r_penalty', 'current_r'}
    assert required == set(losses.keys()), \
        f"Missing: {required - set(losses.keys())}, Extra: {set(losses.keys()) - required}"

    # All losses are scalar (current_r is a float metric, not a loss tensor)
    for name, loss in losses.items():
        if isinstance(loss, float):
            continue  # current_r is a diagnostic metric
        assert loss.shape == (), f"{name} is not scalar"
        assert not torch.isnan(loss), f"{name} is NaN"


def test_loss_backward(model, mnist_batch):
    """Gradients flow through total loss."""
    output = model(mnist_batch)
    target = torch.randint(0, 10, (4,))
    losses = model.compute_loss(output, target)
    losses['total'].backward()

    # Check gradients exist on key parameters (excluding Hebbian weights)
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients after backward"


def test_consciousness_metrics(model, mnist_batch):
    """Consciousness metrics are extractable."""
    output = model(mnist_batch)
    metrics = model.get_consciousness_metrics(output)

    assert 'reflexivity_score' in metrics
    assert 0 < metrics['reflexivity_score'] <= 1.0


def test_parameter_count(model):
    """Parameter counting per module works."""
    counts = model.count_parameters()
    assert 'total' in counts
    assert 'v1' in counts
    assert 'consciousness' in counts
    assert counts['total'] > 0
    # Sum of parts equals total
    part_sum = sum(v for k, v in counts.items() if k != 'total')
    assert counts['total'] == part_sum, \
        f"Total {counts['total']} != sum of parts {part_sum}"


def test_config_stored(model):
    """Model config is stored for reproducibility."""
    assert 'num_classes' in model.config
    assert model.config['num_classes'] == 10
    assert model.config['consciousness_dim'] == 32


def test_cifar_input():
    """Model handles CIFAR-10 sized input (32x32)."""
    from src.model import SubstanceNet
    model = SubstanceNet(num_classes=10, v1_scales=4, v1_dim=128)
    if hasattr(model, 'v3') and hasattr(model.v3, 'output_proj'):
        model.v3.output_proj.set_learning(False)
    if hasattr(model, 'v4'):
        for attr in ['compress_in', 'compress_out']:
            if hasattr(model.v4, attr):
                getattr(model.v4, attr).set_learning(False)
    x = torch.randn(2, 1, 32, 32)
    output = model(x)
    assert output['logits'].shape == (2, 10)


def test_video_mode(model):
    """Video mode processes frame sequences."""
    from src.data.dynamic_primitives import generate_sequence
    frames, _, _ = generate_sequence(primitive_type=0, num_frames=6,
                                     dx=2.0, noise_std=0.0)
    output = model(frames.unsqueeze(0), mode='video')
    assert 'logits' in output
    assert output['logits'].shape[1] == 10


def test_cognitive_mode(model):
    """Cognitive mode processes flat tensors."""
    x = torch.randn(4, 6)
    output = model(x, mode='cognitive')
    assert 'logits' in output
    assert output['logits'].shape == (4, 10)


def test_reflexivity_in_range(model, mnist_batch):
    """R should be in valid range for fresh model."""
    output = model(mnist_batch)
    metrics = model.get_consciousness_metrics(output)
    r = metrics['reflexivity_score']
    assert 0 < r <= 1.0, f"R={r} out of range"
