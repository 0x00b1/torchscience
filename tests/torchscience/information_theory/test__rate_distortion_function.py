"""Tests for rate-distortion function."""

import pytest
import torch

from torchscience.information import rate_distortion_function


class TestRateDistortionFunctionBasic:
    """Basic functionality tests."""

    def test_output_shape(self):
        """Returns scalar for 1D source distribution."""
        px = torch.tensor([0.5, 0.5])
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = rate_distortion_function(px, d, lagrange_multiplier=1.0)
        assert result.shape == torch.Size([])

    def test_return_distribution(self):
        """Returns tuple when return_distribution=True."""
        px = torch.tensor([0.5, 0.5])
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = rate_distortion_function(
            px, d, lagrange_multiplier=1.0, return_distribution=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        rate, pyx = result
        assert rate.shape == torch.Size([])
        assert pyx.shape == torch.Size([2, 2])


class TestRateDistortionFunctionCorrectness:
    """Numerical correctness tests."""

    def test_rate_increases_with_beta(self):
        """Rate increases as beta increases (lower distortion)."""
        px = torch.tensor([0.5, 0.5])
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        rate_low = rate_distortion_function(px, d, lagrange_multiplier=0.5)
        rate_high = rate_distortion_function(px, d, lagrange_multiplier=5.0)

        assert rate_high > rate_low

    def test_zero_beta_zero_rate(self):
        """At beta=0, rate is approximately 0."""
        px = torch.tensor([0.5, 0.5])
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = rate_distortion_function(
            px, d, lagrange_multiplier=0.0, base=2.0
        )
        assert torch.isclose(result, torch.tensor(0.0), atol=0.1)

    def test_large_beta_high_rate(self):
        """At large beta, rate approaches H(X)."""
        px = torch.tensor([0.5, 0.5])
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = rate_distortion_function(
            px, d, lagrange_multiplier=20.0, base=2.0
        )
        # Should approach H(X) = 1 bit for uniform binary source
        assert result > 0.8

    def test_non_negative_rate(self):
        """Rate is non-negative."""
        px = torch.tensor([0.3, 0.7])
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = rate_distortion_function(px, d, lagrange_multiplier=2.0)
        assert result >= -1e-6

    def test_rate_bounded_by_entropy(self):
        """Rate is bounded by source entropy."""
        from torchscience.information import shannon_entropy

        px = torch.tensor([0.3, 0.7])
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = rate_distortion_function(
            px, d, lagrange_multiplier=10.0, base=2.0
        )
        h_x = shannon_entropy(px, base=2.0)
        assert result <= h_x + 0.1


class TestRateDistortionFunctionBatching:
    """Tests for batched operations."""

    def test_batch_consistency(self):
        """Batched computation matches individual."""
        px = torch.tensor([0.5, 0.5])
        d1 = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        d2 = torch.tensor([[0.0, 2.0], [2.0, 0.0]])  # Scaled distortion
        d_batch = torch.stack([d1, d2])

        r1 = rate_distortion_function(px, d1, lagrange_multiplier=1.0)
        r2 = rate_distortion_function(px, d2, lagrange_multiplier=1.0)
        r_batch = rate_distortion_function(
            px, d_batch, lagrange_multiplier=1.0
        )

        assert torch.isclose(r_batch[0], r1, atol=1e-5)
        assert torch.isclose(r_batch[1], r2, atol=1e-5)


class TestRateDistortionFunctionEdgeCases:
    """Tests for edge cases."""

    def test_single_symbol_source(self):
        """Deterministic source has rate 0."""
        px = torch.tensor([1.0])
        d = torch.tensor([[0.0, 1.0]])  # 1 source, 2 repr
        result = rate_distortion_function(px, d, lagrange_multiplier=1.0)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-5)

    def test_asymmetric_distortion(self):
        """Works with asymmetric distortion matrix."""
        px = torch.tensor([0.5, 0.5])
        d = torch.tensor([[0.0, 0.5], [1.0, 0.0]])  # Asymmetric
        result = rate_distortion_function(px, d, lagrange_multiplier=1.0)
        assert torch.isfinite(result)


class TestRateDistortionFunctionDevice:
    """Tests for device handling."""

    def test_cpu(self):
        """Works on CPU."""
        px = torch.tensor([0.5, 0.5], device="cpu")
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="cpu")
        result = rate_distortion_function(px, d, lagrange_multiplier=1.0)
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda(self):
        """Works on CUDA."""
        px = torch.tensor([0.5, 0.5], device="cuda")
        d = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device="cuda")
        result = rate_distortion_function(px, d, lagrange_multiplier=1.0)
        assert result.device.type == "cuda"
