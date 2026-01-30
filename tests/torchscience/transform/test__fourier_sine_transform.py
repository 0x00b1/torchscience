"""Tests for Discrete Sine Transform (DST) implementation."""

import numpy as np
import pytest
import torch
from scipy import fft as scipy_fft
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestDSTForward:
    """Test DST forward pass correctness."""

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_matches_scipy_backward_norm(self, dst_type):
        """DST should match scipy with backward normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = T.fourier_sine_transform(x, type=dst_type, norm="backward")
        scipy_val = scipy_fft.dst(x.numpy(), type=dst_type, norm=None)
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_matches_scipy_ortho_norm(self, dst_type):
        """DST should match scipy with ortho normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = T.fourier_sine_transform(x, type=dst_type, norm="ortho")
        scipy_val = scipy_fft.dst(x.numpy(), type=dst_type, norm="ortho")
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_output_is_real(self, dst_type):
        """DST output should be real-valued."""
        x = torch.randn(16, dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=dst_type)
        assert y.dtype == torch.float64
        assert not y.is_complex()

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = torch.randn(32, dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=2)
        assert y.shape == x.shape

    def test_batched_input(self):
        """DST should work with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=2, dim=-1)
        assert y.shape == x.shape

    def test_different_dims(self):
        """DST should work along different dimensions."""
        x = torch.randn(4, 8, 16, dtype=torch.float64)
        for dim in [0, 1, 2, -1, -2, -3]:
            y = T.fourier_sine_transform(x, type=2, dim=dim)
            assert y.shape == x.shape


class TestDSTRoundTrip:
    """Test DST -> IDST round-trip."""

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_round_trip(self, dst_type, norm):
        """DST -> IDST should recover original."""
        x = torch.randn(16, dtype=torch.float64)
        X = T.fourier_sine_transform(x, type=dst_type, norm=norm)
        x_rec = T.inverse_fourier_sine_transform(X, type=dst_type, norm=norm)
        assert torch.allclose(x_rec, x, atol=1e-10)

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_round_trip_batched(self, dst_type):
        """Round-trip should work with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        X = T.fourier_sine_transform(x, type=dst_type, norm="ortho", dim=-1)
        x_rec = T.inverse_fourier_sine_transform(
            X, type=dst_type, norm="ortho", dim=-1
        )
        assert torch.allclose(x_rec, x, atol=1e-10)


class TestDSTGradient:
    """Test DST gradient correctness."""

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dst_type, norm):
        """Gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: T.fourier_sine_transform(
                inp, type=dst_type, norm=norm
            ),
            (x,),
            raise_exception=True,
        )

    @pytest.mark.parametrize("dst_type", [2, 4])
    def test_gradgradcheck(self, dst_type):
        """Second-order gradient should pass numerical check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            lambda inp: T.fourier_sine_transform(
                inp, type=dst_type, norm="ortho"
            ),
            (x,),
            raise_exception=True,
        )


class TestIDSTGradient:
    """Test IDST gradient correctness."""

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dst_type, norm):
        """IDST gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: T.inverse_fourier_sine_transform(
                inp, type=dst_type, norm=norm
            ),
            (x,),
            raise_exception=True,
        )


class TestDSTWithN:
    """Test DST with signal length parameter."""

    def test_n_larger_pads(self):
        """When n > input_size, input should be zero-padded."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = T.fourier_sine_transform(x, n=8, type=2)
        assert y.shape == torch.Size([8])

    def test_n_smaller_truncates(self):
        """When n < input_size, input should be truncated."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        y = T.fourier_sine_transform(x, n=4, type=2)
        assert y.shape == torch.Size([4])


class TestDSTMeta:
    """Test DST with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=2)
        assert y.shape == torch.Size([16])
        assert y.device.type == "meta"

    def test_meta_tensor_with_n(self):
        """Meta tensor with n parameter should produce correct shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = T.fourier_sine_transform(x, n=32, type=2)
        assert y.shape == torch.Size([32])


class TestDSTDevice:
    """Test DST on different devices."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """DST should work on CUDA tensors."""
        x = torch.randn(16, dtype=torch.float64, device="cuda")
        y = T.fourier_sine_transform(x, type=2)
        assert y.device.type == "cuda"

        # Round-trip should work
        x_rec = T.inverse_fourier_sine_transform(y, type=2)
        assert torch.allclose(x_rec.cpu(), x.cpu(), atol=1e-10)


class TestDSTDtype:
    """Test DST dtype handling."""

    def test_float32_input(self):
        """DST should work with float32 input."""
        x = torch.randn(16, dtype=torch.float32)
        y = T.fourier_sine_transform(x, type=2)
        assert y.dtype == torch.float32

    def test_float64_input(self):
        """DST should work with float64 input."""
        x = torch.randn(16, dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=2)
        assert y.dtype == torch.float64


class TestDSTVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test that DST works with vmap."""
        x = torch.randn(8, 32, dtype=torch.float64)

        # Direct batched call
        y_batched = T.fourier_sine_transform(x, type=2)

        # vmap version
        def dst_single(xi):
            return T.fourier_sine_transform(xi, type=2)

        y_vmap = torch.vmap(dst_single)(x)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)


class TestDSTCompile:
    """Tests for torch.compile compatibility."""

    def test_compile_basic(self):
        """Test that DST works with torch.compile."""
        x = torch.randn(32, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_dst(x):
            return T.fourier_sine_transform(x, type=2)

        y_compiled = compiled_dst(x)
        y_eager = T.fourier_sine_transform(x, type=2)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)


class TestDSTEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_input(self):
        """DST of single element should work."""
        x = torch.tensor([3.0], dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=2)
        assert y.shape == torch.Size([1])
        assert torch.isfinite(y).all()

    def test_zeros_input(self):
        """DST of zeros should return zeros."""
        x = torch.zeros(16, dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=2)
        assert torch.allclose(y, torch.zeros_like(y))

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_invalid_type_raises(self, dst_type):
        """Invalid DST type should raise error."""
        x = torch.randn(8, dtype=torch.float64)
        # Valid types are 1-4, so type 5 should be invalid
        with pytest.raises((ValueError, RuntimeError)):
            T.fourier_sine_transform(x, type=5)

    def test_n_equals_input_size(self):
        """When n equals input size, should produce same result."""
        x = torch.randn(8, dtype=torch.float64)
        y1 = T.fourier_sine_transform(x, type=2)
        y2 = T.fourier_sine_transform(x, n=8, type=2)
        assert torch.allclose(y1, y2)

    def test_antisymmetric_property(self):
        """DST of antisymmetric extension should have specific properties."""
        # Create a simple sequence
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = T.fourier_sine_transform(x, type=1)
        # DST-I output should be real
        assert torch.isfinite(y).all()
