"""Tests for FFT-based convolution implementation."""

import numpy as np
import pytest
import torch
from scipy import signal as scipy_signal
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestConvolutionForward:
    """Test convolution forward pass correctness."""

    def test_full_mode_matches_scipy(self):
        """Convolution in full mode should match scipy."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        h = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)

        our = T.convolution(x, h, mode="full")
        scipy_val = scipy_signal.convolve(x.numpy(), h.numpy(), mode="full")

        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    def test_same_mode_matches_scipy(self):
        """Convolution in same mode should match scipy."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        h = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)

        our = T.convolution(x, h, mode="same")
        scipy_val = scipy_signal.convolve(x.numpy(), h.numpy(), mode="same")

        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    def test_valid_mode_matches_scipy(self):
        """Convolution in valid mode should match scipy."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        h = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)

        our = T.convolution(x, h, mode="valid")
        scipy_val = scipy_signal.convolve(x.numpy(), h.numpy(), mode="valid")

        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    def test_output_shape_full(self):
        """Full mode should have output length N + M - 1."""
        x = torch.randn(10, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)

        y = T.convolution(x, h, mode="full")
        assert y.shape == torch.Size([10 + 5 - 1])

    def test_output_shape_same(self):
        """Same mode should have output length N."""
        x = torch.randn(10, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)

        y = T.convolution(x, h, mode="same")
        assert y.shape == torch.Size([10])

    def test_output_shape_valid(self):
        """Valid mode should have output length N - M + 1."""
        x = torch.randn(10, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)

        y = T.convolution(x, h, mode="valid")
        assert y.shape == torch.Size([10 - 5 + 1])

    def test_batched_input(self):
        """Convolution should work with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        h = torch.randn(4, dtype=torch.float64)

        y = T.convolution(x, h, dim=-1)
        assert y.shape == torch.Size([3, 5, 16 + 4 - 1])

    def test_different_dims(self):
        """Convolution should work along different dimensions."""
        x = torch.randn(8, 16, dtype=torch.float64)
        h = torch.randn(3, dtype=torch.float64)

        y0 = T.convolution(x, h, dim=0)
        assert y0.shape == torch.Size([8 + 3 - 1, 16])

        y1 = T.convolution(x, h, dim=1)
        assert y1.shape == torch.Size([8, 16 + 3 - 1])

    def test_identity_kernel(self):
        """Convolution with delta kernel should return the input."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        h = torch.tensor([1.0], dtype=torch.float64)

        y = T.convolution(x, h, mode="same")
        assert torch.allclose(y, x, atol=1e-10)

    def test_commutativity(self):
        """Convolution should be commutative."""
        x = torch.randn(10, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)

        y1 = T.convolution(x, h, mode="full")
        y2 = T.convolution(h, x, mode="full")

        assert torch.allclose(y1, y2, atol=1e-10)


class TestConvolutionGradient:
    """Test convolution gradient correctness."""

    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_gradcheck_input(self, mode):
        """Gradient w.r.t. input should pass numerical check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        h = torch.randn(3, dtype=torch.float64)

        assert gradcheck(
            lambda inp: T.convolution(inp, h, mode=mode),
            (x,),
            raise_exception=True,
        )

    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_gradcheck_kernel(self, mode):
        """Gradient w.r.t. kernel should pass numerical check."""
        x = torch.randn(8, dtype=torch.float64)
        h = torch.randn(3, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda kernel: T.convolution(x, kernel, mode=mode),
            (h,),
            raise_exception=True,
        )

    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_gradcheck_both(self, mode):
        """Gradient w.r.t. both input and kernel should pass."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        h = torch.randn(3, dtype=torch.float64, requires_grad=True)

        assert gradcheck(
            lambda inp, kernel: T.convolution(inp, kernel, mode=mode),
            (x, h),
            raise_exception=True,
        )

    def test_gradgradcheck(self):
        """Second-order gradient should pass numerical check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        h = torch.randn(3, dtype=torch.float64, requires_grad=True)

        assert gradgradcheck(
            lambda inp, kernel: T.convolution(inp, kernel, mode="full"),
            (x, h),
            raise_exception=True,
        )


class TestConvolutionMeta:
    """Test convolution with meta tensors."""

    @pytest.mark.parametrize("mode", ["full", "same", "valid"])
    def test_meta_tensor_shape(self, mode):
        """Meta tensor should produce correct output shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        h = torch.empty(5, device="meta", dtype=torch.float64)

        y = T.convolution(x, h, mode=mode)

        if mode == "full":
            expected = 16 + 5 - 1
        elif mode == "same":
            expected = 16
        else:  # valid
            expected = 16 - 5 + 1

        assert y.shape == torch.Size([expected])
        assert y.device.type == "meta"


class TestConvolutionDevice:
    """Test convolution on different devices."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Convolution should work on CUDA tensors."""
        x = torch.randn(16, dtype=torch.float64, device="cuda")
        h = torch.randn(5, dtype=torch.float64, device="cuda")

        y = T.convolution(x, h, mode="full")
        assert y.device.type == "cuda"
        assert y.shape == torch.Size([16 + 5 - 1])


class TestConvolutionDtype:
    """Test convolution dtype handling."""

    def test_float32_input(self):
        """Convolution should work with float32 input."""
        x = torch.randn(16, dtype=torch.float32)
        h = torch.randn(5, dtype=torch.float32)
        y = T.convolution(x, h, mode="full")
        assert y.dtype == torch.float32

    def test_float64_input(self):
        """Convolution should work with float64 input."""
        x = torch.randn(16, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)
        y = T.convolution(x, h, mode="full")
        assert y.dtype == torch.float64


class TestConvolutionVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test that convolution works with vmap."""
        x = torch.randn(8, 32, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)

        # Direct batched call
        y_batched = T.convolution(x, h, mode="full")

        # vmap version
        def conv_single(xi):
            return T.convolution(xi, h, mode="full")

        y_vmap = torch.vmap(conv_single)(x)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)


class TestConvolutionCompile:
    """Tests for torch.compile compatibility."""

    def test_compile_basic(self):
        """Test that convolution works with torch.compile."""
        x = torch.randn(32, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_conv(x, h):
            return T.convolution(x, h, mode="full")

        y_compiled = compiled_conv(x, h)
        y_eager = T.convolution(x, h, mode="full")

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)


class TestConvolutionEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_mode_raises(self):
        """Should raise error for invalid mode."""
        x = torch.randn(10, dtype=torch.float64)
        h = torch.randn(3, dtype=torch.float64)

        with pytest.raises(ValueError, match="mode"):
            T.convolution(x, h, mode="invalid")

    def test_single_element_input(self):
        """Should work with single element input."""
        x = torch.tensor([3.0], dtype=torch.float64)
        h = torch.tensor([2.0], dtype=torch.float64)

        y = T.convolution(x, h, mode="full")
        assert torch.allclose(y, torch.tensor([6.0], dtype=torch.float64))

    def test_single_element_kernel(self):
        """Convolution with single element kernel should scale."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        h = torch.tensor([2.0], dtype=torch.float64)

        y = T.convolution(x, h, mode="same")
        assert torch.allclose(y, x * 2.0)

    def test_kernel_longer_than_input_valid_raises(self):
        """Valid mode with kernel longer than input should raise error."""
        x = torch.randn(3, dtype=torch.float64)
        h = torch.randn(5, dtype=torch.float64)

        # Valid mode requires input >= kernel length
        with pytest.raises(RuntimeError, match="valid.*mode.*input.*kernel"):
            T.convolution(x, h, mode="valid")

    def test_zeros_input(self):
        """Convolution of zeros should return zeros."""
        x = torch.zeros(10, dtype=torch.float64)
        h = torch.randn(3, dtype=torch.float64)

        y = T.convolution(x, h, mode="full")
        assert torch.allclose(y, torch.zeros_like(y))

    def test_zeros_kernel(self):
        """Convolution with zero kernel should return zeros."""
        x = torch.randn(10, dtype=torch.float64)
        h = torch.zeros(3, dtype=torch.float64)

        y = T.convolution(x, h, mode="full")
        assert torch.allclose(y, torch.zeros_like(y))
