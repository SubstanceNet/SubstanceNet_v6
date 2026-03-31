"""
Tests for feature projection (v6).
Replaces QuantumWaveFunction tests after v6 cleanup.
"""
import pytest
import torch
from src.model.substance_net import SubstanceNet


@pytest.fixture
def model():
    return SubstanceNet(num_classes=10)


def test_feature_proj_output_shape(model):
    """Feature projection produces correct shape."""
    x = torch.randn(2, 1, 28, 28)
    out = model(x, mode='image')
    # amplitude and phase should have same shape
    assert out['amplitude'].shape == out['phase'].shape
    assert out['amplitude'].shape[0] == 2  # batch
    assert out['amplitude'].shape[-1] == 64  # half of wave_channels=128


def test_amplitude_non_negative(model):
    """Amplitude-like features are non-negative (ReLU)."""
    x = torch.randn(4, 1, 28, 28)
    out = model(x, mode='image')
    # After ReLU, features should be >= 0
    features = torch.cat([out['amplitude'], out['phase']], dim=-1)
    assert (features >= 0).all()


def test_feature_regularization_computable(model):
    """Feature regularization loss is computable."""
    x = torch.randn(2, 1, 28, 28)
    out = model(x, mode='image')
    reg = 0.01 * (out['amplitude'].pow(2).mean() +
                   out['phase'].pow(2).mean())
    assert reg.item() >= 0
    assert not torch.isnan(reg)


def test_feature_dim_consistency(model):
    """Feature dimensions consistent across modes."""
    img = torch.randn(2, 1, 28, 28)
    cog = torch.randn(2, 3, 3)
    out_img = model(img, mode='image')
    out_cog = model(cog, mode='cognitive')
    assert out_img['amplitude'].shape[-1] == out_cog['amplitude'].shape[-1]
