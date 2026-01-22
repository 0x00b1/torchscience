"""Tests for performance features: torch.compile and vmap compatibility."""

import pytest
import torch

from torchscience.spline import (
    b_spline_evaluate,
    b_spline_fit,
    cubic_spline_evaluate,
    cubic_spline_fit,
    pchip_evaluate,
    pchip_fit,
)


class TestTorchCompile:
    """Tests for torch.compile compatibility."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_cubic_spline_evaluate_compile(self):
        """Cubic spline evaluation should work with torch.compile."""
        x = torch.linspace(0, 1, 10)
        y = torch.sin(x * torch.pi)
        spline = cubic_spline_fit(x, y)

        # Compile the evaluation function
        @torch.compile
        def evaluate(t):
            return cubic_spline_evaluate(spline, t)

        t = torch.linspace(0.1, 0.9, 5)
        result = evaluate(t)

        # Should produce same result as non-compiled
        expected = cubic_spline_evaluate(spline, t)
        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_b_spline_evaluate_compile(self):
        """B-spline evaluation should work with torch.compile."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        @torch.compile
        def evaluate(t):
            return b_spline_evaluate(spline, t)

        t = torch.linspace(0.1, 0.9, 5)
        result = evaluate(t)

        expected = b_spline_evaluate(spline, t)
        assert torch.allclose(result, expected, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_pchip_evaluate_compile(self):
        """PCHIP evaluation should work with torch.compile."""
        x = torch.linspace(0, 1, 10)
        y = torch.sin(x * torch.pi)
        spline = pchip_fit(x, y)

        @torch.compile
        def evaluate(t):
            return pchip_evaluate(spline, t)

        t = torch.linspace(0.1, 0.9, 5)
        result = evaluate(t)

        expected = pchip_evaluate(spline, t)
        assert torch.allclose(result, expected, atol=1e-5)


class TestVmap:
    """Tests for vmap compatibility."""

    @pytest.mark.xfail(
        reason="vmap not yet supported with tensorclass splines"
    )
    def test_cubic_spline_evaluate_vmap_queries(self):
        """Cubic spline evaluation should support vmap over queries."""
        x = torch.linspace(0, 1, 10)
        y = torch.sin(x * torch.pi)
        spline = cubic_spline_fit(x, y)

        # Batch of query points
        queries = torch.rand(5, 3)  # 5 batches of 3 queries each

        def eval_single(t):
            return cubic_spline_evaluate(spline, t)

        # vmap over batch dimension
        result = torch.vmap(eval_single)(queries)

        assert result.shape == (5, 3)

        # Verify correctness
        for i in range(5):
            expected = cubic_spline_evaluate(spline, queries[i])
            assert torch.allclose(result[i], expected, atol=1e-5)

    @pytest.mark.xfail(
        reason="vmap not yet supported with tensorclass splines"
    )
    def test_b_spline_evaluate_vmap_queries(self):
        """B-spline evaluation should support vmap over queries."""
        x = torch.linspace(0, 1, 10)
        y = x**2
        spline = b_spline_fit(x, y, degree=3)

        queries = torch.rand(5, 3)

        def eval_single(t):
            return b_spline_evaluate(spline, t)

        result = torch.vmap(eval_single)(queries)

        assert result.shape == (5, 3)

    @pytest.mark.xfail(
        reason="vmap not yet supported with tensorclass splines"
    )
    def test_pchip_evaluate_vmap_queries(self):
        """PCHIP evaluation should support vmap over queries."""
        x = torch.linspace(0, 1, 10)
        y = torch.sin(x * torch.pi)
        spline = pchip_fit(x, y)

        queries = torch.rand(5, 3)

        def eval_single(t):
            return pchip_evaluate(spline, t)

        result = torch.vmap(eval_single)(queries)

        assert result.shape == (5, 3)


class TestBatchedOperations:
    """Tests for efficient batched operations."""

    def test_batched_cubic_spline_evaluation(self):
        """Batched evaluation should be efficient."""
        x = torch.linspace(0, 1, 100)
        y = torch.sin(x * 2 * torch.pi)
        spline = cubic_spline_fit(x, y)

        # Large batch of queries
        t = torch.rand(10000)

        # Single call should handle all queries
        result = cubic_spline_evaluate(spline, t)

        assert result.shape == (10000,)

    def test_batched_b_spline_evaluation(self):
        """Batched B-spline evaluation should be efficient."""
        x = torch.linspace(0, 1, 100)
        y = torch.sin(x * 2 * torch.pi)
        spline = b_spline_fit(x, y, degree=3)

        t = torch.rand(10000)
        result = b_spline_evaluate(spline, t)

        assert result.shape == (10000,)

    def test_multidimensional_query_shapes(self):
        """Should handle arbitrary query shapes."""
        x = torch.linspace(0, 1, 20)
        y = x**2
        spline = cubic_spline_fit(x, y)

        # Various shapes
        for shape in [(10,), (5, 5), (2, 3, 4), (2, 2, 2, 2)]:
            t = torch.rand(shape)
            result = cubic_spline_evaluate(spline, t)
            assert result.shape == shape


class TestGradientPerformance:
    """Tests for efficient gradient computation."""

    def test_cubic_spline_backward_efficiency(self):
        """Gradient computation should be efficient."""
        x = torch.linspace(0, 1, 100)
        y = torch.sin(x * 2 * torch.pi)
        spline = cubic_spline_fit(x, y, extrapolate="clamp")

        t = torch.rand(1000, requires_grad=True)
        result = cubic_spline_evaluate(spline, t)
        loss = result.sum()
        loss.backward()

        assert t.grad is not None
        assert t.grad.shape == (1000,)

    def test_b_spline_backward_efficiency(self):
        """B-spline gradient computation should be efficient."""
        x = torch.linspace(0, 1, 100)
        y = torch.sin(x * 2 * torch.pi)
        spline = b_spline_fit(x, y, degree=3, extrapolate="clamp")

        t = torch.rand(1000, requires_grad=True)
        result = b_spline_evaluate(spline, t)
        loss = result.sum()
        loss.backward()

        assert t.grad is not None
        assert t.grad.shape == (1000,)

    def test_second_order_gradients(self):
        """Second-order gradients should work."""
        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = x**2
        spline = cubic_spline_fit(x, y, extrapolate="clamp")

        t = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

        # First derivative
        result = cubic_spline_evaluate(spline, t)
        grad1 = torch.autograd.grad(result, t, create_graph=True)[0]

        # Second derivative
        grad2 = torch.autograd.grad(grad1, t)[0]

        assert grad2 is not None
