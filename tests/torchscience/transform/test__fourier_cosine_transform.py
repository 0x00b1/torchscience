"""Tests for Discrete Cosine Transform (DCT) implementation."""

import numpy as np
import pytest
import torch
from scipy import fft as scipy_fft
from torch.autograd import gradcheck, gradgradcheck

import torchscience.transform as T


class TestDCTForward:
    """Test DCT forward pass correctness."""

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_matches_scipy_backward_norm(self, dct_type):
        """DCT should match scipy with backward normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = T.fourier_cosine_transform(x, type=dct_type, norm="backward")
        scipy_val = scipy_fft.dct(x.numpy(), type=dct_type, norm=None)
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_matches_scipy_ortho_norm(self, dct_type):
        """DCT should match scipy with ortho normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = T.fourier_cosine_transform(x, type=dct_type, norm="ortho")
        scipy_val = scipy_fft.dct(x.numpy(), type=dct_type, norm="ortho")
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_output_is_real(self, dct_type):
        """DCT output should be real-valued."""
        x = torch.randn(16, dtype=torch.float64)
        y = T.fourier_cosine_transform(x, type=dct_type)
        assert y.dtype == torch.float64
        assert not y.is_complex()

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = torch.randn(32, dtype=torch.float64)
        y = T.fourier_cosine_transform(x, type=2)
        assert y.shape == x.shape

    def test_batched_input(self):
        """DCT should work with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        y = T.fourier_cosine_transform(x, type=2, dim=-1)
        assert y.shape == x.shape

    def test_different_dims(self):
        """DCT should work along different dimensions."""
        x = torch.randn(4, 8, 16, dtype=torch.float64)
        for dim in [0, 1, 2, -1, -2, -3]:
            y = T.fourier_cosine_transform(x, type=2, dim=dim)
            assert y.shape == x.shape


class TestDCTRoundTrip:
    """Test DCT -> IDCT round-trip."""

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_round_trip(self, dct_type, norm):
        """DCT -> IDCT should recover original."""
        x = torch.randn(16, dtype=torch.float64)
        X = T.fourier_cosine_transform(x, type=dct_type, norm=norm)
        x_rec = T.inverse_fourier_cosine_transform(X, type=dct_type, norm=norm)
        assert torch.allclose(x_rec, x, atol=1e-10)

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_round_trip_batched(self, dct_type):
        """Round-trip should work with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        X = T.fourier_cosine_transform(x, type=dct_type, norm="ortho", dim=-1)
        x_rec = T.inverse_fourier_cosine_transform(
            X, type=dct_type, norm="ortho", dim=-1
        )
        assert torch.allclose(x_rec, x, atol=1e-10)


class TestDCTGradient:
    """Test DCT gradient correctness."""

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dct_type, norm):
        """Gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: T.fourier_cosine_transform(
                inp, type=dct_type, norm=norm
            ),
            (x,),
            raise_exception=True,
        )

    @pytest.mark.parametrize("dct_type", [2, 4])
    def test_gradgradcheck(self, dct_type):
        """Second-order gradient should pass numerical check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            lambda inp: T.fourier_cosine_transform(
                inp, type=dct_type, norm="ortho"
            ),
            (x,),
            raise_exception=True,
        )


class TestIDCTGradient:
    """Test IDCT gradient correctness."""

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dct_type, norm):
        """IDCT gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: T.inverse_fourier_cosine_transform(
                inp, type=dct_type, norm=norm
            ),
            (x,),
            raise_exception=True,
        )


class TestDCTWithN:
    """Test DCT with signal length parameter."""

    def test_n_larger_pads(self):
        """When n > input_size, input should be zero-padded."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = T.fourier_cosine_transform(x, n=8, type=2)
        assert y.shape == torch.Size([8])

    def test_n_smaller_truncates(self):
        """When n < input_size, input should be truncated."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        y = T.fourier_cosine_transform(x, n=4, type=2)
        assert y.shape == torch.Size([4])


class TestDCTMeta:
    """Test DCT with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = T.fourier_cosine_transform(x, type=2)
        assert y.shape == torch.Size([16])
        assert y.device.type == "meta"

    def test_meta_tensor_with_n(self):
        """Meta tensor with n parameter should produce correct shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = T.fourier_cosine_transform(x, n=32, type=2)
        assert y.shape == torch.Size([32])


class TestDCTDevice:
    """Test DCT on different devices."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """DCT should work on CUDA tensors."""
        x = torch.randn(16, dtype=torch.float64, device="cuda")
        y = T.fourier_cosine_transform(x, type=2)
        assert y.device.type == "cuda"

        # Round-trip should work
        x_rec = T.inverse_fourier_cosine_transform(y, type=2)
        assert torch.allclose(x_rec.cpu(), x.cpu(), atol=1e-10)


class TestDCTVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test that DCT works with vmap."""
        x = torch.randn(8, 32, dtype=torch.float64)

        # Direct batched call
        y_batched = T.fourier_cosine_transform(x, type=2)

        # vmap version
        def dct_single(xi):
            return T.fourier_cosine_transform(xi, type=2)

        y_vmap = torch.vmap(dct_single)(x)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)


class TestDCTCompile:
    """Tests for torch.compile compatibility."""

    def test_compile_basic(self):
        """Test that DCT works with torch.compile."""
        x = torch.randn(32, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_dct(x):
            return T.fourier_cosine_transform(x, type=2)

        y_compiled = compiled_dct(x)
        y_eager = T.fourier_cosine_transform(x, type=2)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)


class TestDCTAutocast:
    """Tests for autocast compatibility."""

    @pytest.mark.skip(
        reason="fourier_cosine_transform C++ backend segfaults with float16 inputs"
    )
    @pytest.mark.parametrize("dct_type", [2, 4])
    def test_autocast_cpu_float16(self, dct_type):
        """Test that DCT works under CPU autocast with float16."""
        x = torch.randn(32, dtype=torch.float16)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            y = T.fourier_cosine_transform(x, type=dct_type)

        # Should produce valid output
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    @pytest.mark.skip(
        reason="fourier_cosine_transform C++ backend segfaults with bfloat16 inputs"
    )
    @pytest.mark.parametrize("dct_type", [2, 4])
    def test_autocast_cpu_bfloat16(self, dct_type):
        """Test that DCT works under CPU autocast with bfloat16."""
        x = torch.randn(32, dtype=torch.bfloat16)

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            y = T.fourier_cosine_transform(x, type=dct_type)

        assert y.shape == x.shape
        assert not torch.isnan(y).any()
