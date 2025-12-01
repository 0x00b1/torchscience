"""Tests for chi_squared_divergence operator."""

import pytest
import torch

from torchscience.information_theory import chi_squared_divergence


class TestChiSquaredDivergenceBasic:
    """Basic functionality tests."""

    def test_output_shape_1d(self):
        """Output is scalar for 1D input."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)
        result = chi_squared_divergence(p, q)
        assert result.shape == torch.Size([])

    def test_output_shape_2d_batch(self):
        """Output has batch dims for 2D input."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result = chi_squared_divergence(p, q)
        assert result.shape == torch.Size([5])

    def test_output_shape_3d_batch(self):
        """Output has batch dims for 3D input."""
        p = torch.softmax(torch.randn(3, 4, 10), dim=-1)
        q = torch.softmax(torch.randn(3, 4, 10), dim=-1)
        result = chi_squared_divergence(p, q)
        assert result.shape == torch.Size([3, 4])


class TestChiSquaredDivergenceCorrectness:
    """Correctness tests."""

    def test_identical_distributions_zero(self):
        """Chi-squared divergence is 0 for identical distributions."""
        p = torch.softmax(torch.randn(10), dim=-1)
        result = chi_squared_divergence(p, p)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_uniform_identical_zero(self):
        """Chi-squared divergence is 0 for identical uniform distributions."""
        p = torch.ones(8) / 8
        result = chi_squared_divergence(p, p)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_non_negative(self):
        """Chi-squared divergence is always non-negative."""
        torch.manual_seed(42)
        for _ in range(10):
            p = torch.softmax(torch.randn(10), dim=-1)
            q = torch.softmax(torch.randn(10), dim=-1)
            result = chi_squared_divergence(p, q)
            assert result >= -1e-6

    def test_known_value(self):
        """Test against known computed value."""
        # p = [0.25, 0.25, 0.25, 0.25], q = [0.5, 0.25, 0.125, 0.125]
        # χ² = (0.25-0.5)²/0.5 + (0.25-0.25)²/0.25 + (0.25-0.125)²/0.125 + (0.25-0.125)²/0.125
        # χ² = 0.0625/0.5 + 0 + 0.015625/0.125 + 0.015625/0.125
        # χ² = 0.125 + 0 + 0.125 + 0.125 = 0.375... wait let me recalculate
        # χ² = (0.25-0.5)²/0.5 + 0 + (0.125)²/0.125 + (0.125)²/0.125
        # χ² = (-0.25)²/0.5 + (0.125)²/0.125 + (0.125)²/0.125
        # χ² = 0.0625/0.5 + 0.015625/0.125 + 0.015625/0.125
        # χ² = 0.125 + 0.125 + 0.125 = 0.375
        # But example in docstring says 0.75... let me verify
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.5, 0.25, 0.125, 0.125])
        result = chi_squared_divergence(p, q)
        # Actually: (0.25-0.5)²/0.5 + 0 + (0.25-0.125)²/0.125 + (0.25-0.125)²/0.125
        # = 0.0625/0.5 + 0.015625/0.125 + 0.015625/0.125
        # = 0.125 + 0.125 + 0.125 = 0.375
        # Hmm, the docstring example has 0.75, that's different...
        # Actually should be using χ²(P||Q) = sum (p-q)²/q
        # = (0.25-0.5)²/0.5 + (0.25-0.25)²/0.25 + (0.25-0.125)²/0.125 + (0.25-0.125)²/0.125
        # = 0.125 + 0 + 0.125 + 0.125 = 0.375
        expected = torch.tensor(0.375)
        assert torch.isclose(result, expected, rtol=1e-4)


class TestChiSquaredDivergenceReduction:
    """Reduction mode tests."""

    def test_reduction_none(self):
        """Reduction 'none' preserves batch dimensions."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result = chi_squared_divergence(p, q, reduction="none")
        assert result.shape == torch.Size([5])

    def test_reduction_sum(self):
        """Reduction 'sum' returns scalar."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result = chi_squared_divergence(p, q, reduction="sum")
        assert result.shape == torch.Size([])

    def test_reduction_mean(self):
        """Reduction 'mean' returns scalar."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result = chi_squared_divergence(p, q, reduction="mean")
        assert result.shape == torch.Size([])

    def test_reduction_sum_equals_manual(self):
        """Reduction 'sum' equals manual sum."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result_none = chi_squared_divergence(p, q, reduction="none")
        result_sum = chi_squared_divergence(p, q, reduction="sum")
        assert torch.isclose(result_sum, result_none.sum(), rtol=1e-5)

    def test_reduction_mean_equals_manual(self):
        """Reduction 'mean' equals manual mean."""
        p = torch.softmax(torch.randn(5, 10), dim=-1)
        q = torch.softmax(torch.randn(5, 10), dim=-1)
        result_none = chi_squared_divergence(p, q, reduction="none")
        result_mean = chi_squared_divergence(p, q, reduction="mean")
        assert torch.isclose(result_mean, result_none.mean(), rtol=1e-5)


class TestChiSquaredDivergenceGradients:
    """Gradient tests."""

    def test_gradcheck(self):
        """First-order gradients pass gradcheck."""
        p = torch.softmax(
            torch.randn(5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)
        q = torch.softmax(
            torch.randn(5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)

        def func(p, q):
            return chi_squared_divergence(p, q)

        assert torch.autograd.gradcheck(
            func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradgradcheck(self):
        """Second-order gradients pass gradgradcheck."""
        p = torch.softmax(
            torch.randn(5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)
        q = torch.softmax(
            torch.randn(5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)

        def func(p, q):
            return chi_squared_divergence(p, q)

        assert torch.autograd.gradgradcheck(
            func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_batched(self):
        """First-order gradients pass gradcheck for batched input."""
        p = torch.softmax(
            torch.randn(3, 5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)
        q = torch.softmax(
            torch.randn(3, 5, dtype=torch.float64), dim=-1
        ).requires_grad_(True)

        def func(p, q):
            return chi_squared_divergence(p, q, reduction="sum")

        assert torch.autograd.gradcheck(
            func, (p, q), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_backward_runs(self):
        """Backward pass runs without error."""
        p = torch.softmax(torch.randn(10), dim=-1).requires_grad_(True)
        q = torch.softmax(torch.randn(10), dim=-1).requires_grad_(True)
        result = chi_squared_divergence(p, q)
        result.backward()
        assert p.grad is not None
        assert q.grad is not None


class TestChiSquaredDivergenceEdgeCases:
    """Edge case tests."""

    def test_near_zero_q(self):
        """Handles near-zero q values without NaN."""
        p = torch.tensor([0.5, 0.5, 0.0, 0.0])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = chi_squared_divergence(p, q)
        assert not torch.isnan(result)
        assert not torch.isinf(result)

    def test_dtype_float32(self):
        """Works with float32."""
        p = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float32), dim=-1)
        result = chi_squared_divergence(p, q)
        assert result.dtype == torch.float32

    def test_dtype_float64(self):
        """Works with float64."""
        p = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(10, dtype=torch.float64), dim=-1)
        result = chi_squared_divergence(p, q)
        assert result.dtype == torch.float64


class TestChiSquaredDivergenceValidation:
    """Input validation tests."""

    def test_invalid_p_type(self):
        """Raises error for non-Tensor p."""
        q = torch.softmax(torch.randn(10), dim=-1)
        with pytest.raises(TypeError, match="p must be a Tensor"):
            chi_squared_divergence([0.5, 0.5], q)

    def test_invalid_q_type(self):
        """Raises error for non-Tensor q."""
        p = torch.softmax(torch.randn(10), dim=-1)
        with pytest.raises(TypeError, match="q must be a Tensor"):
            chi_squared_divergence(p, [0.5, 0.5])

    def test_shape_mismatch(self):
        """Raises error for shape mismatch."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(8), dim=-1)
        with pytest.raises(ValueError, match="must have the same shape"):
            chi_squared_divergence(p, q)

    def test_invalid_reduction(self):
        """Raises error for invalid reduction."""
        p = torch.softmax(torch.randn(10), dim=-1)
        q = torch.softmax(torch.randn(10), dim=-1)
        with pytest.raises(ValueError, match="reduction must be one of"):
            chi_squared_divergence(p, q, reduction="invalid")


class TestChiSquaredDivergenceDim:
    """Dimension parameter tests."""

    def test_dim_0(self):
        """Compute along dim 0."""
        p = torch.softmax(torch.randn(10, 5), dim=0)
        q = torch.softmax(torch.randn(10, 5), dim=0)
        result = chi_squared_divergence(p, q, dim=0)
        assert result.shape == torch.Size([5])

    def test_dim_negative(self):
        """Negative dim works correctly."""
        p = torch.softmax(torch.randn(3, 4, 5), dim=-1)
        q = torch.softmax(torch.randn(3, 4, 5), dim=-1)
        result = chi_squared_divergence(p, q, dim=-1)
        assert result.shape == torch.Size([3, 4])


class TestChiSquaredDivergenceMeta:
    """Meta tensor tests."""

    def test_meta_tensor_shape(self):
        """Meta tensor shape inference is correct."""
        p = torch.softmax(torch.randn(5, 10), dim=-1).to("meta")
        q = torch.softmax(torch.randn(5, 10), dim=-1).to("meta")
        result = chi_squared_divergence(p, q)
        assert result.shape == torch.Size([5])
        assert result.device.type == "meta"
