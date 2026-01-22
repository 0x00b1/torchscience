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


class TestAllOperatorsAutocast:
    """Autocast tests for all differentiation operators."""

    def test_derivative_autocast(self):
        from torchscience.differentiation import derivative

        field = torch.randn(16, 16, dtype=torch.float16)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = derivative(field, dim=-1, dx=0.1)
        assert result.dtype == torch.float32

    def test_laplacian_autocast(self):
        from torchscience.differentiation import laplacian

        field = torch.randn(16, 16, dtype=torch.float16)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = laplacian(field, dx=0.1)
        assert result.dtype == torch.float32

    def test_hessian_autocast(self):
        from torchscience.differentiation import hessian

        field = torch.randn(16, 16, dtype=torch.float16)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = hessian(field, dx=0.1)
        assert result.dtype == torch.float32

    def test_biharmonic_autocast(self):
        from torchscience.differentiation import biharmonic

        field = torch.randn(16, 16, dtype=torch.float16)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = biharmonic(field, dx=0.1)
        assert result.dtype == torch.float32

    def test_divergence_autocast(self):
        from torchscience.differentiation import divergence

        field = torch.randn(2, 16, 16, dtype=torch.float16)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = divergence(field, dx=0.1)
        assert result.dtype == torch.float32

    def test_curl_autocast(self):
        from torchscience.differentiation import curl

        field = torch.randn(3, 8, 8, 8, dtype=torch.float16)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = curl(field, dx=0.1)
        assert result.dtype == torch.float32

    def test_jacobian_autocast(self):
        from torchscience.differentiation import jacobian

        field = torch.randn(3, 8, 8, 8, dtype=torch.float16)
        with torch.amp.autocast("cpu", dtype=torch.float16):
            result = jacobian(field, dx=0.1)
        assert result.dtype == torch.float32


class TestAllOperatorsAutograd:
    """Autograd tests for all differentiation operators."""

    def test_derivative_gradcheck(self):
        from torchscience.differentiation import derivative

        field = torch.randn(12, 12, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda f: derivative(f, dim=-1, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_derivative_gradgradcheck(self):
        from torchscience.differentiation import derivative

        field = torch.randn(12, 12, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            lambda f: derivative(f, dim=-1, dx=0.1).sum(),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_laplacian_gradcheck(self):
        from torchscience.differentiation import laplacian

        field = torch.randn(12, 12, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda f: laplacian(f, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_laplacian_gradgradcheck(self):
        from torchscience.differentiation import laplacian

        field = torch.randn(12, 12, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            lambda f: laplacian(f, dx=0.1).sum(),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_divergence_gradcheck(self):
        from torchscience.differentiation import divergence

        field = torch.randn(2, 12, 12, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda f: divergence(f, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_curl_gradcheck(self):
        from torchscience.differentiation import curl

        field = torch.randn(
            3, 8, 8, 8, dtype=torch.float64, requires_grad=True
        )
        assert gradcheck(
            lambda f: curl(f, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_hessian_gradcheck(self):
        from torchscience.differentiation import hessian

        field = torch.randn(10, 10, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda f: hessian(f, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_jacobian_gradcheck(self):
        from torchscience.differentiation import jacobian

        field = torch.randn(2, 10, 10, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda f: jacobian(f, dx=0.1),
            (field,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
        )
