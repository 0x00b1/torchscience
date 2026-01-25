"""Tests for rate-distortion Lagrangian."""

import math

import pytest
import torch

from torchscience.information.compression import (
    estimate_bitrate,
    rate_distortion_lagrangian,
)


class TestRateDistortionLagrangianBasic:
    """Basic functionality tests."""

    def test_output_type(self):
        """Returns tensor."""
        rate = torch.tensor(100.0)
        distortion = torch.tensor(0.01)
        J = rate_distortion_lagrangian(rate, distortion)
        assert isinstance(J, torch.Tensor)

    def test_output_is_scalar(self):
        """Returns scalar for scalar inputs."""
        rate = torch.tensor(100.0)
        distortion = torch.tensor(0.01)
        J = rate_distortion_lagrangian(rate, distortion)
        assert J.dim() == 0


class TestRateDistortionLagrangianFormula:
    """Tests for formula correctness."""

    def test_formula_r_plus_lambda_d(self):
        """J = R + λD."""
        rate = torch.tensor(100.0)
        distortion = torch.tensor(0.5)
        lmbda = 0.1

        J = rate_distortion_lagrangian(rate, distortion, lmbda=lmbda)
        expected = rate + lmbda * distortion
        assert torch.isclose(J, expected)

    def test_zero_distortion(self):
        """Zero distortion gives J = R."""
        rate = torch.tensor(50.0)
        distortion = torch.tensor(0.0)
        J = rate_distortion_lagrangian(rate, distortion, lmbda=0.1)
        assert torch.isclose(J, rate)

    def test_zero_rate(self):
        """Zero rate gives J = λD."""
        rate = torch.tensor(0.0)
        distortion = torch.tensor(10.0)
        lmbda = 0.5
        J = rate_distortion_lagrangian(rate, distortion, lmbda=lmbda)
        expected = lmbda * distortion
        assert torch.isclose(J, expected)

    def test_lambda_scaling(self):
        """Higher λ increases distortion weight."""
        rate = torch.tensor(100.0)
        distortion = torch.tensor(1.0)

        J_low = rate_distortion_lagrangian(rate, distortion, lmbda=0.01)
        J_high = rate_distortion_lagrangian(rate, distortion, lmbda=1.0)

        assert J_high > J_low


class TestRateDistortionLagrangianBatched:
    """Tests for batched inputs."""

    def test_batched_rate(self):
        """Works with batched rate."""
        rate = torch.tensor([100.0, 200.0, 300.0])
        distortion = torch.tensor(0.1)
        J = rate_distortion_lagrangian(rate, distortion, lmbda=0.1)
        assert J.shape == rate.shape

    def test_batched_distortion(self):
        """Works with batched distortion."""
        rate = torch.tensor(100.0)
        distortion = torch.tensor([0.1, 0.2, 0.3])
        J = rate_distortion_lagrangian(rate, distortion, lmbda=0.1)
        assert J.shape == distortion.shape

    def test_batched_both(self):
        """Works with both batched."""
        rate = torch.tensor([100.0, 200.0])
        distortion = torch.tensor([0.1, 0.2])
        J = rate_distortion_lagrangian(rate, distortion, lmbda=0.1)
        assert J.shape == (2,)


class TestRateDistortionLagrangianGradients:
    """Tests for gradient computation."""

    def test_gradient_to_rate(self):
        """Gradients flow to rate."""
        rate = torch.tensor(100.0, requires_grad=True)
        distortion = torch.tensor(0.1)
        J = rate_distortion_lagrangian(rate, distortion)
        J.backward()
        assert rate.grad is not None
        assert torch.isclose(rate.grad, torch.tensor(1.0))  # dJ/dR = 1

    def test_gradient_to_distortion(self):
        """Gradients flow to distortion."""
        rate = torch.tensor(100.0)
        distortion = torch.tensor(0.1, requires_grad=True)
        lmbda = 0.5
        J = rate_distortion_lagrangian(rate, distortion, lmbda=lmbda)
        J.backward()
        assert distortion.grad is not None
        assert torch.isclose(distortion.grad, torch.tensor(lmbda))  # dJ/dD = λ


class TestEstimateBitrateBasic:
    """Basic tests for estimate_bitrate."""

    def test_output_type(self):
        """Returns tensor."""
        likelihoods = torch.tensor([0.5, 0.5])
        bits = estimate_bitrate(likelihoods)
        assert isinstance(bits, torch.Tensor)

    def test_uniform_distribution(self):
        """Uniform over N gives log2(N) bits each."""
        N = 8
        likelihoods = torch.ones(N) / N
        bits = estimate_bitrate(likelihoods)
        # Each symbol: -log2(1/8) = 3 bits, total = 8 * 3 = 24
        expected = N * math.log2(N)
        assert torch.isclose(bits, torch.tensor(expected), rtol=1e-5)

    def test_certain_symbol_zero_bits(self):
        """Probability 1 gives 0 bits."""
        likelihoods = torch.tensor([1.0])
        bits = estimate_bitrate(likelihoods)
        assert torch.isclose(bits, torch.tensor(0.0))


class TestEstimateBitrateFormula:
    """Tests for bitrate formula correctness."""

    def test_sum_of_bits(self):
        """Sum of bits for observed symbols."""
        # Two symbols with probabilities 0.25 and 0.75
        likelihoods = torch.tensor([0.25, 0.75])
        bits = estimate_bitrate(likelihoods, reduction="sum")
        # Total bits = -log2(0.25) + -log2(0.75) = 2 + 0.415 = 2.415
        expected = -math.log2(0.25) - math.log2(0.75)
        assert torch.isclose(bits, torch.tensor(expected), rtol=1e-5)

    def test_bits_per_symbol(self):
        """Mean reduction gives bits per symbol."""
        N = 16
        likelihoods = torch.ones(N) / N
        bits_per_symbol = estimate_bitrate(likelihoods, reduction="mean")
        # Each symbol: -log2(1/16) = 4 bits
        expected = math.log2(N)
        assert torch.isclose(
            bits_per_symbol, torch.tensor(expected), rtol=1e-5
        )

    def test_no_reduction(self):
        """None reduction returns per-element bits."""
        likelihoods = torch.tensor([0.5, 0.25, 0.25])
        bits = estimate_bitrate(likelihoods, reduction="none")
        assert bits.shape == likelihoods.shape
        # Check individual bits
        assert torch.isclose(
            bits[0], torch.tensor(1.0), rtol=1e-5
        )  # -log2(0.5) = 1
        assert torch.isclose(
            bits[1], torch.tensor(2.0), rtol=1e-5
        )  # -log2(0.25) = 2


class TestEstimateBitrateGradients:
    """Tests for bitrate gradients."""

    def test_gradient_exists(self):
        """Gradients flow through bitrate."""
        likelihoods = torch.tensor([0.3, 0.7], requires_grad=True)
        bits = estimate_bitrate(likelihoods)
        bits.backward()
        assert likelihoods.grad is not None


class TestEstimateBitrateEdgeCases:
    """Edge case tests."""

    def test_very_small_probability(self):
        """Handles very small probabilities."""
        likelihoods = torch.tensor([1e-20])
        bits = estimate_bitrate(likelihoods)
        # Should not be inf due to clamping
        assert not torch.isinf(bits)
        assert bits > 0

    def test_not_tensor_raises(self):
        """Raises error for non-tensor input."""
        with pytest.raises(TypeError, match="must be a Tensor"):
            estimate_bitrate([0.5, 0.5])

    def test_invalid_reduction_raises(self):
        """Raises error for invalid reduction."""
        likelihoods = torch.tensor([0.5, 0.5])
        with pytest.raises(ValueError, match="reduction"):
            estimate_bitrate(likelihoods, reduction="invalid")
