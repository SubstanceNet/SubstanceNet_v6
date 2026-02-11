"""
System Classification: tests.test_model
Author: Oleksii Onasenko
Developer: SubstanceNet — https://github.com/SubstanceNet
Code: Claude (Anthropic)
License: MIT

Tests for SubstanceNet model template.
"""

import torch
import pytest


@pytest.fixture
def model():
    from src.model import SubstanceNet
    return SubstanceNet(num_classes=10)


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

    expected = {'total', 'classification', 'consciousness',
                'zero_loss', 'phase_coherence', 'topological'}
    assert expected == set(losses.keys())

    # All losses are scalar
    for name, loss in losses.items():
        assert loss.shape == (), f"{name} is not scalar"
        assert not torch.isnan(loss), f"{name} is NaN"


def test_loss_backward(model, mnist_batch):
    """Gradients flow through total loss."""
    output = model(mnist_batch)
    target = torch.randint(0, 10, (4,))
    losses = model.compute_loss(output, target)
    losses['total'].backward()

    # Check gradients exist on key parameters
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
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
    assert counts['total'] == sum(
        v for k, v in counts.items() if k != 'total')


def test_config_stored(model):
    """Model config is stored for reproducibility."""
    assert 'num_classes' in model.config
    assert model.config['num_classes'] == 10
    assert model.config['consciousness_dim'] == 32


def test_cifar_input():
    """Model handles CIFAR-10 sized input (32x32)."""
    from src.model import SubstanceNet
    model = SubstanceNet(num_classes=10, v1_scales=4, v1_dim=128)
    x = torch.randn(2, 1, 32, 32)
    output = model(x)
    assert output['logits'].shape == (2, 10)
