"""Comprehensive tests for f-divergence."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.information import (
    chi_squared_divergence,
    f_divergence,
    kullback_leibler_divergence,
)


class TestFDivergenceBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Returns scalar for 1D probability vectors."""
        p = torch.tensor([0.4, 0.3, 0.2, 0.1])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])

        def f(t):
            return t * torch.log(t)

        result = f_divergence(p, q, f)
        assert result.shape == torch.Size([])

    def test_output_shape_batch(self):
        """Returns correct shape for batched input."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        def f(t):
            return t * torch.log(t)

        result = f_divergence(p, q, f)
        assert result.shape == torch.Size([10])

    def test_output_shape_3d(self):
        """Returns correct shape for 3D batched input."""
        p = torch.softmax(torch.randn(4, 8, 6), dim=-1)
        q = torch.softmax(torch.randn(4, 8, 6), dim=-1)

        def f(t):
            return (t - 1) ** 2

        result = f_divergence(p, q, f)
        assert result.shape == torch.Size([4, 8])


class TestFDivergenceCorrectness:
    """Numerical correctness tests."""

    def test_matches_kl_divergence(self):
        """f(t) = t*log(t) gives KL divergence."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def kl_generator(t):
            return t * torch.log(t)

        result = f_divergence(p, q, kl_generator)
        expected = kullback_leibler_divergence(p, q)

        assert torch.isclose(result, expected, rtol=1e-4)

    def test_matches_chi_squared(self):
        """f(t) = (t-1)^2 gives chi-squared divergence."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def chi_squared_generator(t):
            return (t - 1) ** 2

        result = f_divergence(p, q, chi_squared_generator)
        expected = chi_squared_divergence(p, q)

        assert torch.isclose(result, expected, rtol=1e-4)

    def test_zero_for_identical(self):
        """f(1) = 0 implies D_f(P||P) = 0."""
        p = torch.softmax(torch.randn(10), dim=-1)

        def f(t):
            return (t - 1) ** 2

        result = f_divergence(p, p, f)

        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_non_negativity_kl_generator(self):
        """f-divergence with KL generator is non-negative."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def kl_f(t):
            return t * torch.log(t)

        result = f_divergence(p, q, kl_f)
        assert result >= -1e-6

    def test_non_negativity_chi_squared_generator(self):
        """f-divergence with chi-squared generator is non-negative."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def chi_f(t):
            return (t - 1) ** 2

        result = f_divergence(p, q, chi_f)
        assert result >= -1e-6

    def test_reverse_kl_generator(self):
        """f(t) = -log(t) gives reverse KL divergence."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def reverse_kl_generator(t):
            return -torch.log(t)

        result = f_divergence(p, q, reverse_kl_generator)
        # Reverse KL: D_KL(Q || P) = sum_i q_i * log(q_i / p_i) = -sum_i q_i * log(p_i / q_i)
        # The f-divergence formula: sum_i q_i * f(p_i / q_i) = sum_i q_i * (-log(p_i/q_i))
        # = sum_i q_i * log(q_i / p_i) = D_KL(Q || P)
        expected = kullback_leibler_divergence(q, p)

        assert torch.isclose(result, expected, rtol=1e-4)

    def test_hellinger_generator(self):
        """f(t) = (sqrt(t) - 1)^2 gives squared Hellinger distance."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def hellinger_generator(t):
            return (torch.sqrt(t) - 1) ** 2

        result = f_divergence(p, q, hellinger_generator)

        # Squared Hellinger: H^2(P, Q) = sum_i (sqrt(p_i) - sqrt(q_i))^2 / 2
        # = 1 - sum_i sqrt(p_i * q_i)
        # f-divergence: sum_i q_i * (sqrt(p_i/q_i) - 1)^2
        # = sum_i q_i * (p_i/q_i - 2*sqrt(p_i/q_i) + 1)
        # = sum_i p_i - 2*sum_i sqrt(p_i*q_i) + sum_i q_i
        # = 2 - 2*sum_i sqrt(p_i*q_i) = 2 * H^2(P, Q)
        hellinger_sq = 1 - torch.sum(torch.sqrt(p * q))
        expected = 2 * hellinger_sq

        assert torch.isclose(result, expected, rtol=1e-4)

    def test_total_variation_generator(self):
        """f(t) = |t-1|/2 gives total variation distance."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def tv_generator(t):
            return torch.abs(t - 1) / 2

        result = f_divergence(p, q, tv_generator)

        # Total variation: TV(P, Q) = (1/2) * sum_i |p_i - q_i|
        expected = torch.sum(torch.abs(p - q)) / 2

        assert torch.isclose(result, expected, rtol=1e-4)


class TestFDivergenceInputTypes:
    """Input type handling tests."""

    def test_log_probability_input(self):
        """Works with log probability inputs."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)

        def f(t):
            return t * torch.log(t)

        # Direct probability input
        result_prob = f_divergence(p, q, f, input_type="probability")

        # Log probability input
        result_log = f_divergence(
            torch.log(p), torch.log(q), f, input_type="log_probability"
        )

        assert torch.isclose(result_prob, result_log, rtol=1e-4)

    def test_logits_input(self):
        """Works with logits inputs."""
        logits_p = torch.randn(10)
        logits_q = torch.randn(10)

        def f(t):
            return t * torch.log(t)

        # Logits input
        result_logits = f_divergence(
            logits_p, logits_q, f, input_type="logits"
        )

        # Manual softmax
        p = torch.softmax(logits_p, dim=-1)
        q = torch.softmax(logits_q, dim=-1)
        result_prob = f_divergence(p, q, f, input_type="probability")

        assert torch.isclose(result_logits, result_prob, rtol=1e-4)


class TestFDivergenceReductions:
    """Reduction option tests."""

    def test_reduction_none(self):
        """reduction='none' returns per-sample divergences."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        def f(t):
            return t * torch.log(t)

        result = f_divergence(p, q, f, reduction="none")
        assert result.shape == torch.Size([10])

    def test_reduction_sum(self):
        """reduction='sum' returns sum of divergences."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        def f(t):
            return t * torch.log(t)

        result_none = f_divergence(p, q, f, reduction="none")
        result_sum = f_divergence(p, q, f, reduction="sum")

        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)

    def test_reduction_mean(self):
        """reduction='mean' returns mean of divergences."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        def f(t):
            return t * torch.log(t)

        result_none = f_divergence(p, q, f, reduction="none")
        result_mean = f_divergence(p, q, f, reduction="mean")

        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)

    def test_reduction_batchmean(self):
        """reduction='batchmean' returns mean over batch dimension."""
        p = torch.softmax(torch.randn(10, 5), dim=-1)
        q = torch.softmax(torch.randn(10, 5), dim=-1)

        def f(t):
            return t * torch.log(t)

        result_none = f_divergence(p, q, f, reduction="none")
        result_batchmean = f_divergence(p, q, f, reduction="batchmean")

        expected = result_none.sum() / result_none.numel()
        assert torch.isclose(result_batchmean, expected, rtol=1e-5)


class TestFDivergenceDimension:
    """Dimension handling tests."""

    def test_dim_0(self):
        """Works with dim=0."""
        p = torch.softmax(torch.randn(5, 10), dim=0)
        q = torch.softmax(torch.randn(5, 10), dim=0)

        def f(t):
            return t * torch.log(t)

        result = f_divergence(p, q, f, dim=0)
        assert result.shape == torch.Size([10])

    def test_dim_negative(self):
        """Works with negative dim."""
        p = torch.softmax(torch.randn(4, 8, 6), dim=-1)
        q = torch.softmax(torch.randn(4, 8, 6), dim=-1)

        def f(t):
            return t * torch.log(t)

        result_neg = f_divergence(p, q, f, dim=-1)
        result_pos = f_divergence(p, q, f, dim=2)

        assert torch.allclose(result_neg, result_pos, rtol=1e-5)


class TestFDivergenceGradients:
    """Gradient computation tests."""

    def test_gradcheck_kl(self):
        """Gradients are correct with KL generator."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def f(t):
            return t * torch.log(t)

        def func(p_in, q_in):
            return f_divergence(p_in, q_in, f)

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_chi_squared(self):
        """Gradients are correct with chi-squared generator."""
        p = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def f(t):
            return (t - 1) ** 2

        def func(p_in, q_in):
            return f_divergence(p_in, q_in, f)

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradcheck_batched(self):
        """Gradients are correct for batched inputs."""
        p = torch.softmax(torch.randn(4, 5, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(4, 5, dtype=torch.float64), dim=-1)
        p.requires_grad_(True)
        q.requires_grad_(True)

        def f(t):
            return t * torch.log(t)

        def func(p_in, q_in):
            return f_divergence(p_in, q_in, f, reduction="sum")

        assert gradcheck(func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestFDivergenceValidation:
    """Input validation tests."""

    def test_non_tensor_p(self):
        """Raises error for non-tensor p."""

        def f(t):
            return t * torch.log(t)

        with pytest.raises(TypeError, match="p must be a Tensor"):
            f_divergence([0.5, 0.5], torch.tensor([0.5, 0.5]), f)

    def test_non_tensor_q(self):
        """Raises error for non-tensor q."""

        def f(t):
            return t * torch.log(t)

        with pytest.raises(TypeError, match="q must be a Tensor"):
            f_divergence(torch.tensor([0.5, 0.5]), [0.5, 0.5], f)

    def test_non_callable_f(self):
        """Raises error for non-callable f."""
        p = torch.tensor([0.5, 0.5])
        q = torch.tensor([0.5, 0.5])

        with pytest.raises(TypeError, match="f must be a callable"):
            f_divergence(p, q, "not a function")

    def test_invalid_input_type(self):
        """Raises error for invalid input_type."""

        def f(t):
            return t * torch.log(t)

        with pytest.raises(ValueError, match="input_type must be one of"):
            f_divergence(
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.5, 0.5]),
                f,
                input_type="invalid",
            )

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""

        def f(t):
            return t * torch.log(t)

        with pytest.raises(ValueError, match="reduction must be one of"):
            f_divergence(
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.5, 0.5]),
                f,
                reduction="invalid",
            )

    def test_invalid_dim(self):
        """Raises error for invalid dim."""

        def f(t):
            return t * torch.log(t)

        with pytest.raises(IndexError, match="dim .* out of range"):
            f_divergence(
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.5, 0.5]),
                f,
                dim=5,
            )

    def test_mismatched_distribution_sizes(self):
        """Raises error for mismatched distribution sizes."""

        def f(t):
            return t * torch.log(t)

        with pytest.raises(ValueError, match="Distribution sizes must match"):
            f_divergence(
                torch.tensor([0.5, 0.5]),
                torch.tensor([0.33, 0.33, 0.34]),
                f,
            )


class TestFDivergenceDtypes:
    """Data type handling tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Result dtype matches input dtype."""
        p = torch.softmax(torch.randn(10, dtype=dtype), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=dtype), dim=-1)

        def f(t):
            return t * torch.log(t)

        result = f_divergence(p, q, f)
        assert result.dtype == dtype

    def test_dtype_promotion(self):
        """Different input dtypes are promoted."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)

        def f(t):
            return t * torch.log(t)

        result = f_divergence(p, q, f)
        assert result.dtype == torch.float64
