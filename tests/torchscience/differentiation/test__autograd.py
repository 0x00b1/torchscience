"""Tests for autograd support of differentiation operators."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.differentiation import gradient


class TestGradientAutograd:
    """Tests for gradient autograd support."""

    def test_gradient_gradcheck(self):
        """Gradient passes torch.autograd.gradcheck."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        def grad_fn(f):
            return gradient(f, dx=0.1, accuracy=2, boundary="replicate")

        assert gradcheck(grad_fn, (field,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradient_gradgradcheck(self):
        """Gradient passes torch.autograd.gradgradcheck."""
        field = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        def grad_fn(f):
            g = gradient(f, dx=0.1, accuracy=2, boundary="replicate")
            return g.sum()

        assert gradgradcheck(grad_fn, (field,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestGradientVmap:
    """Tests for gradient vmap support."""

    def test_gradient_vmap_batch(self):
        """Gradient works with torch.vmap over batch dimension."""
        batch_fields = torch.randn(4, 16, 16)

        # vmap over batch dimension
        batched_gradient = torch.vmap(lambda f: gradient(f, dx=0.1), in_dims=0)

        result = batched_gradient(batch_fields)

        # Each field is 16x16, gradient adds 2 components
        assert result.shape == (4, 2, 16, 16)

        # Compare with manual loop
        manual = torch.stack(
            [gradient(batch_fields[i], dx=0.1) for i in range(4)]
        )
        torch.testing.assert_close(result, manual)


class TestGradientCompile:
    """Tests for gradient torch.compile support."""

    @pytest.mark.xfail(
        reason="Phase 1: derivative uses tensorclass which causes graph breaks"
    )
    def test_gradient_compile_no_graph_breaks(self):
        """Gradient compiles without graph breaks."""

        @torch.compile(fullgraph=True)
        def compiled_gradient(field):
            return gradient(field, dx=0.1, boundary="replicate")

        field = torch.randn(16, 16)
        result = compiled_gradient(field)

        assert result.shape == (2, 16, 16)

        # Compare with eager
        eager_result = gradient(field, dx=0.1, boundary="replicate")
        torch.testing.assert_close(result, eager_result)

    @pytest.mark.xfail(
        reason="Phase 1: derivative uses tensorclass which causes graph breaks"
    )
    def test_gradient_compile_different_boundaries(self):
        """Gradient compiles for each boundary mode."""
        for boundary in ["replicate", "zeros", "reflect", "circular"]:

            @torch.compile(fullgraph=True)
            def compiled_gradient(field):
                return gradient(field, dx=0.1, boundary=boundary)

            field = torch.randn(16, 16)
            result = compiled_gradient(field)
            assert result.shape == (2, 16, 16)


class TestGradientAutocast:
    """Tests for gradient autocast support."""

    def test_gradient_autocast_fp16(self):
        """Gradient upcasts to fp32 under autocast."""
        field = torch.randn(16, 16, dtype=torch.float16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = gradient(field, dx=0.1)

        # Result should be fp32 (upcasted for numerical stability)
        assert result.dtype == torch.float32
        assert result.shape == (2, 16, 16)

    def test_gradient_autocast_bf16(self):
        """Gradient upcasts to fp32 under bfloat16 autocast."""
        field = torch.randn(16, 16, dtype=torch.bfloat16, device="cpu")

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            result = gradient(field, dx=0.1)

        # Result should be fp32
        assert result.dtype == torch.float32
