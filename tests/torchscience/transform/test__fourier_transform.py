"""Tests for fourier_transform and inverse_fourier_transform."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import fourier_transform, inverse_fourier_transform


class TestFourierTransformForward:
    """Tests for fourier_transform forward pass."""

    def test_basic_real_input(self):
        """Test basic FFT of real input."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        X = fourier_transform(x)

        expected = torch.fft.fft(x)
        assert torch.allclose(X, expected)

    def test_complex_input(self):
        """Test FFT of complex input."""
        x = torch.randn(32, dtype=torch.complex64)
        X = fourier_transform(x)

        expected = torch.fft.fft(x)
        assert torch.allclose(X, expected)

    def test_output_is_complex(self):
        """FFT output should always be complex."""
        x = torch.randn(32)
        X = fourier_transform(x)

        assert X.is_complex()

    def test_shape_preserved(self):
        """Output shape matches input shape along transform dim."""
        x = torch.randn(10, 20, 30)
        X = fourier_transform(x, dim=1)

        assert X.shape == x.shape

    def test_with_n_larger(self):
        """Test padding when n > input size."""
        x = torch.randn(64)
        X = fourier_transform(x, n=128)

        assert X.shape == torch.Size([128])

    def test_with_n_smaller(self):
        """Test truncation when n < input size."""
        x = torch.randn(128)
        X = fourier_transform(x, n=64)

        assert X.shape == torch.Size([64])

    def test_matches_torch_fft(self):
        """Should match torch.fft.fft for various inputs."""
        for shape in [(32,), (16, 32), (8, 16, 32)]:
            x = torch.randn(shape)
            X = fourier_transform(x)
            expected = torch.fft.fft(x)
            assert torch.allclose(X, expected), f"Mismatch for shape {shape}"


class TestFourierTransformNormalization:
    """Tests for normalization modes."""

    def test_backward_norm(self):
        """Test backward normalization."""
        x = torch.randn(32)
        X = fourier_transform(x, norm="backward")
        expected = torch.fft.fft(x, norm="backward")
        assert torch.allclose(X, expected)

    def test_ortho_norm(self):
        """Test ortho normalization."""
        x = torch.randn(32)
        X = fourier_transform(x, norm="ortho")
        expected = torch.fft.fft(x, norm="ortho")
        assert torch.allclose(X, expected)

    def test_forward_norm(self):
        """Test forward normalization."""
        x = torch.randn(32)
        X = fourier_transform(x, norm="forward")
        expected = torch.fft.fft(x, norm="forward")
        assert torch.allclose(X, expected)


class TestFourierTransformPadding:
    """Tests for padding modes."""

    def test_constant_padding(self):
        """Test constant (zero) padding."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="constant")
        assert X.shape == torch.Size([64])

    def test_reflect_padding(self):
        """Test reflect padding."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="reflect")
        assert X.shape == torch.Size([64])

    def test_replicate_padding(self):
        """Test replicate padding."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="replicate")
        assert X.shape == torch.Size([64])

    def test_circular_padding(self):
        """Test circular padding."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="circular")
        assert X.shape == torch.Size([64])

    def test_invalid_padding_mode(self):
        """Test that invalid padding mode raises error."""
        x = torch.randn(32)
        with pytest.raises(ValueError, match="padding_mode"):
            fourier_transform(x, n=64, padding_mode="invalid")


class TestFourierTransformWindow:
    """Tests for windowing."""

    def test_with_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(32)
        window = torch.hann_window(32)
        X = fourier_transform(x, window=window)

        expected = torch.fft.fft(x * window)
        assert torch.allclose(X, expected)

    def test_with_hamming_window(self):
        """Test with Hamming window."""
        x = torch.randn(32)
        window = torch.hamming_window(32)
        X = fourier_transform(x, window=window)

        expected = torch.fft.fft(x * window)
        assert torch.allclose(X, expected)


class TestFourierTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_real_input(self):
        """Test gradient correctness for real input."""
        x = torch.randn(16, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            fourier_transform, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_complex_input(self):
        """Test gradient correctness for complex input."""
        x = torch.randn(16, dtype=torch.complex128, requires_grad=True)
        assert gradcheck(
            fourier_transform, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            fourier_transform, (x,), eps=1e-6, atol=1e-3, rtol=1e-2
        )


class TestInverseFourierTransformForward:
    """Tests for inverse_fourier_transform forward pass."""

    def test_basic_complex_input(self):
        """Test basic IFFT of complex input."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X)

        expected = torch.fft.ifft(X)
        assert torch.allclose(x, expected)

    def test_round_trip(self):
        """Test that FFT followed by IFFT recovers original."""
        x_orig = torch.randn(32)
        X = fourier_transform(x_orig)
        x_rec = inverse_fourier_transform(X)

        assert torch.allclose(x_rec.real, x_orig, atol=1e-5)


class TestInverseFourierTransformGradient:
    """Tests for inverse_fourier_transform gradient computation."""

    def test_gradcheck(self):
        """Test gradient correctness."""
        X = torch.randn(16, dtype=torch.complex128, requires_grad=True)
        assert gradcheck(
            inverse_fourier_transform, (X,), eps=1e-6, atol=1e-4, rtol=1e-3
        )


class TestFourierTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(32, device="cuda")
        X = fourier_transform(x)

        expected = torch.fft.fft(x)
        assert torch.allclose(X, expected)
        assert X.device == x.device


class TestFourierTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.randn(32, device="meta")
        X = fourier_transform(x)

        assert X.shape == torch.Size([32])
        assert X.device == torch.device("meta")

    def test_meta_tensor_with_n(self):
        """Test shape inference with n parameter."""
        x = torch.randn(32, device="meta")
        X = fourier_transform(x, n=64)

        assert X.shape == torch.Size([64])
