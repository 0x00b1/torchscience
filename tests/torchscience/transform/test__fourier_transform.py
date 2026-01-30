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


class TestFourierTransformDtype:
    """Test Fourier transform dtype handling."""

    def test_float32_input(self):
        """FFT should work with float32 input."""
        x = torch.randn(32, dtype=torch.float32)
        X = fourier_transform(x)
        assert X.dtype == torch.complex64

    def test_float64_input(self):
        """FFT should work with float64 input."""
        x = torch.randn(32, dtype=torch.float64)
        X = fourier_transform(x)
        assert X.dtype == torch.complex128

    def test_complex64_input(self):
        """FFT should work with complex64 input."""
        x = torch.randn(32, dtype=torch.complex64)
        X = fourier_transform(x)
        assert X.dtype == torch.complex64

    def test_complex128_input(self):
        """FFT should work with complex128 input."""
        x = torch.randn(32, dtype=torch.complex128)
        X = fourier_transform(x)
        assert X.dtype == torch.complex128


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


class TestFourierTransformMultiDim:
    """Tests for multi-dimensional transform support."""

    def test_2d_transform(self):
        """Test 2D FFT with dim tuple."""
        x = torch.randn(8, 16, 32)
        X = fourier_transform(x, dim=(-2, -1))
        expected = torch.fft.fft2(x, dim=(-2, -1))
        assert torch.allclose(X, expected)

    def test_nd_transform(self):
        """Test nD FFT with dim tuple."""
        x = torch.randn(4, 8, 16)
        X = fourier_transform(x, dim=(0, 1, 2))
        expected = torch.fft.fftn(x)
        assert torch.allclose(X, expected)

    def test_n_with_multi_dim(self):
        """Test n parameter with multi-dim."""
        x = torch.randn(16, 16)
        X = fourier_transform(x, dim=(-2, -1), n=(32, 32))
        assert X.shape == torch.Size([32, 32])

    def test_n_tuple_truncation(self):
        """Test n parameter for truncation with multi-dim."""
        x = torch.randn(32, 32)
        X = fourier_transform(x, dim=(-2, -1), n=(16, 16))
        assert X.shape == torch.Size([16, 16])

    def test_single_dim_as_tuple(self):
        """Test single dim provided as tuple."""
        x = torch.randn(32, 64)
        X = fourier_transform(x, dim=(-1,))
        expected = torch.fft.fft(x, dim=-1)
        assert torch.allclose(X, expected)

    def test_n_tuple_length_mismatch_raises(self):
        """Test that mismatched n and dim tuple lengths raise error."""
        x = torch.randn(16, 16)
        with pytest.raises(ValueError, match="length"):
            fourier_transform(x, dim=(-2, -1), n=(32,))

    def test_2d_with_batch_dim(self):
        """Test 2D FFT preserves batch dimensions."""
        x = torch.randn(4, 8, 16, 32)
        X = fourier_transform(x, dim=(-2, -1))
        expected = torch.fft.fft2(x, dim=(-2, -1))
        assert torch.allclose(X, expected)
        assert X.shape == x.shape


class TestFourierTransformNewPadding:
    """Tests for new padding modes from torchscience.pad."""

    def test_linear_padding(self):
        """Test linear extrapolation padding."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="linear")
        assert X.shape == torch.Size([64])

    def test_smooth_padding(self):
        """Test smooth padding mode."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="smooth")
        assert X.shape == torch.Size([64])

    def test_polynomial_padding(self):
        """Test polynomial padding mode."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="polynomial")
        assert X.shape == torch.Size([64])

    def test_spline_padding(self):
        """Test spline padding mode."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="spline")
        assert X.shape == torch.Size([64])

    def test_padding_order_for_polynomial(self):
        """Test padding_order parameter for polynomial mode."""
        x = torch.randn(32)
        X = fourier_transform(
            x, n=64, padding_mode="polynomial", padding_order=2
        )
        assert X.shape == torch.Size([64])

    def test_padding_order_quadratic(self):
        """Test quadratic polynomial padding."""
        x = torch.randn(32)
        X1 = fourier_transform(
            x, n=64, padding_mode="polynomial", padding_order=1
        )
        X2 = fourier_transform(
            x, n=64, padding_mode="polynomial", padding_order=2
        )
        # Different orders should give different results (usually)
        # Just check they both work and have correct shape
        assert X1.shape == torch.Size([64])
        assert X2.shape == torch.Size([64])

    def test_reflect_odd_padding(self):
        """Test reflect_odd (antisymmetric) padding mode."""
        x = torch.randn(32)
        X = fourier_transform(x, n=64, padding_mode="reflect_odd")
        assert X.shape == torch.Size([64])

    def test_new_padding_modes_with_multi_dim(self):
        """Test new padding modes work with multi-dim transforms."""
        x = torch.randn(16, 16)
        X = fourier_transform(
            x, dim=(-2, -1), n=(32, 32), padding_mode="linear"
        )
        assert X.shape == torch.Size([32, 32])


class TestFourierTransformExplicitPadding:
    """Tests for explicit padding parameter."""

    def test_explicit_padding_1d(self):
        """Test explicit padding for 1D."""
        x = torch.randn(32)
        # Pad 8 on left, 8 on right
        X = fourier_transform(x, padding=(8, 8))
        assert X.shape == torch.Size([48])

    def test_explicit_padding_asymmetric(self):
        """Test asymmetric explicit padding."""
        x = torch.randn(32)
        X = fourier_transform(x, padding=(4, 12))
        assert X.shape == torch.Size([48])

    def test_explicit_padding_multi_dim(self):
        """Test explicit padding for multi-dim transform."""
        x = torch.randn(16, 16)
        # Pad each dim by (4, 4)
        X = fourier_transform(x, dim=(-2, -1), padding=((4, 4), (4, 4)))
        assert X.shape == torch.Size([24, 24])

    def test_n_overrides_padding(self):
        """Test that n parameter works with explicit padding."""
        x = torch.randn(32)
        # Explicit padding of 8 each side would give 48, but n=64 should override
        X = fourier_transform(x, padding=(8, 8), n=64)
        assert X.shape == torch.Size([64])


class TestFourierTransformParameterOrder:
    """Tests for new parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        x = torch.randn(32)
        # This should work
        X = fourier_transform(x, dim=-1, n=64, norm="ortho")
        assert X.shape == torch.Size([64])

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        x = torch.randn(32)
        # This should fail - dim should be keyword only
        with pytest.raises(TypeError):
            fourier_transform(x, -1)  # type: ignore


class TestInverseFourierTransformMultiDim:
    """Tests for multi-dimensional inverse transform."""

    def test_2d_inverse(self):
        """Test 2D IFFT with dim tuple."""
        X = torch.randn(16, 32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, dim=(-2, -1))
        expected = torch.fft.ifft2(X)
        assert torch.allclose(x, expected)

    def test_round_trip_2d(self):
        """Test 2D round trip."""
        x_orig = torch.randn(16, 32)
        X = fourier_transform(x_orig, dim=(-2, -1))
        x_rec = inverse_fourier_transform(X, dim=(-2, -1))
        assert torch.allclose(x_rec.real, x_orig, atol=1e-5)

    def test_nd_inverse(self):
        """Test nD IFFT with dim tuple."""
        X = torch.randn(4, 8, 16, dtype=torch.complex64)
        x = inverse_fourier_transform(X, dim=(0, 1, 2))
        expected = torch.fft.ifftn(X)
        assert torch.allclose(x, expected)

    def test_n_with_multi_dim_inverse(self):
        """Test n parameter with multi-dim inverse."""
        X = torch.randn(16, 16, dtype=torch.complex64)
        x = inverse_fourier_transform(X, dim=(-2, -1), n=(32, 32))
        assert x.shape == torch.Size([32, 32])

    def test_single_dim_as_tuple_inverse(self):
        """Test single dim provided as tuple for inverse."""
        X = torch.randn(32, 64, dtype=torch.complex64)
        x = inverse_fourier_transform(X, dim=(-1,))
        expected = torch.fft.ifft(X, dim=-1)
        assert torch.allclose(x, expected)

    def test_2d_with_batch_dim_inverse(self):
        """Test 2D IFFT preserves batch dimensions."""
        X = torch.randn(4, 8, 16, 32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, dim=(-2, -1))
        expected = torch.fft.ifft2(X, dim=(-2, -1))
        assert torch.allclose(x, expected)
        assert x.shape == X.shape


class TestInverseFourierTransformNormalization:
    """Tests for inverse transform normalization modes."""

    def test_backward_norm_inverse(self):
        """Test backward normalization for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, norm="backward")
        expected = torch.fft.ifft(X, norm="backward")
        assert torch.allclose(x, expected)

    def test_ortho_norm_inverse(self):
        """Test ortho normalization for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, norm="ortho")
        expected = torch.fft.ifft(X, norm="ortho")
        assert torch.allclose(x, expected)

    def test_forward_norm_inverse(self):
        """Test forward normalization for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, norm="forward")
        expected = torch.fft.ifft(X, norm="forward")
        assert torch.allclose(x, expected)


class TestInverseFourierTransformPadding:
    """Tests for inverse transform padding modes."""

    def test_constant_padding_inverse(self):
        """Test constant padding for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, n=64, padding_mode="constant")
        assert x.shape == torch.Size([64])

    def test_reflect_padding_inverse(self):
        """Test reflect padding for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, n=64, padding_mode="reflect")
        assert x.shape == torch.Size([64])

    def test_linear_padding_inverse(self):
        """Test linear padding for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, n=64, padding_mode="linear")
        assert x.shape == torch.Size([64])

    def test_smooth_padding_inverse(self):
        """Test smooth padding for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, n=64, padding_mode="smooth")
        assert x.shape == torch.Size([64])

    def test_invalid_padding_mode_inverse(self):
        """Test that invalid padding mode raises error for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        with pytest.raises(ValueError, match="padding_mode"):
            inverse_fourier_transform(X, n=64, padding_mode="invalid")


class TestInverseFourierTransformWindow:
    """Tests for inverse transform windowing."""

    def test_with_hann_window_inverse(self):
        """Test with Hann window for inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        window = torch.hann_window(32)
        x = inverse_fourier_transform(X, window=window)

        expected = torch.fft.ifft(X) * window
        assert torch.allclose(x, expected)


class TestInverseFourierTransformExplicitPadding:
    """Tests for inverse transform explicit padding parameter."""

    def test_explicit_padding_1d_inverse(self):
        """Test explicit padding for 1D inverse."""
        X = torch.randn(32, dtype=torch.complex64)
        x = inverse_fourier_transform(X, padding=(8, 8))
        assert x.shape == torch.Size([48])

    def test_explicit_padding_multi_dim_inverse(self):
        """Test explicit padding for multi-dim inverse."""
        X = torch.randn(16, 16, dtype=torch.complex64)
        x = inverse_fourier_transform(
            X, dim=(-2, -1), padding=((4, 4), (4, 4))
        )
        assert x.shape == torch.Size([24, 24])


class TestFourierTransformVmap:
    """Tests for vmap compatibility."""

    def test_vmap_basic(self):
        """Test that FFT works with vmap."""
        x = torch.randn(8, 32)

        # Direct batched call
        y_batched = fourier_transform(x)

        # vmap version
        def fft_single(xi):
            return fourier_transform(xi)

        y_vmap = torch.vmap(fft_single)(x)

        assert torch.allclose(y_batched, y_vmap, atol=1e-6)


class TestFourierTransformCompile:
    """Tests for torch.compile compatibility."""

    def test_compile_basic(self):
        """Test that FFT works with torch.compile."""
        x = torch.randn(32)

        @torch.compile(fullgraph=True)
        def compiled_fft(x):
            return fourier_transform(x)

        y_compiled = compiled_fft(x)
        y_eager = fourier_transform(x)

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)


class TestFourierTransformAutocast:
    """Tests for autocast compatibility."""

    def test_autocast_cpu_float16(self):
        """Test that FFT works under CPU autocast with float16."""
        x = torch.randn(32, dtype=torch.float16)

        with torch.amp.autocast("cpu", dtype=torch.float16):
            y = fourier_transform(x)

        # FFT should preserve or upcast dtype for numerical stability
        assert y.is_complex()
        assert y.shape == x.shape

    def test_autocast_cpu_bfloat16(self):
        """Test that FFT works under CPU autocast with bfloat16."""
        x = torch.randn(32, dtype=torch.bfloat16)

        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            y = fourier_transform(x)

        assert y.is_complex()
        assert y.shape == x.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_autocast_cuda_float16(self):
        """Test that FFT works under CUDA autocast with float16."""
        x = torch.randn(32, dtype=torch.float16, device="cuda")

        with torch.amp.autocast("cuda", dtype=torch.float16):
            y = fourier_transform(x)

        assert y.is_complex()
        assert y.shape == x.shape


class TestFourierTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_input(self):
        """FFT of single element should work."""
        x = torch.tensor([3.0 + 2j], dtype=torch.complex64)
        y = fourier_transform(x)
        assert y.shape == torch.Size([1])
        # FFT of single element is itself
        assert torch.allclose(y, x)

    def test_zeros_input(self):
        """FFT of zeros should return zeros."""
        x = torch.zeros(32, dtype=torch.complex64)
        y = fourier_transform(x)
        assert torch.allclose(y, torch.zeros_like(y))

    def test_real_input_hermitian_output(self):
        """FFT of real input should have Hermitian symmetry."""
        x = torch.randn(32, dtype=torch.float64)
        y = fourier_transform(x)
        # Y[k] = conj(Y[N-k]) for real input
        for k in range(1, 16):
            assert torch.allclose(y[k], y[32 - k].conj(), atol=1e-10)

    def test_constant_input(self):
        """FFT of constant should have energy only at DC."""
        x = torch.ones(32, dtype=torch.float64)
        y = fourier_transform(x)
        # DC component should be N (or normalized)
        assert y[0].abs() > y[1:].abs().max() * 1000

    def test_impulse_input(self):
        """FFT of impulse should be constant."""
        x = torch.zeros(32, dtype=torch.float64)
        x[0] = 1.0
        y = fourier_transform(x)
        # All frequency components should be equal (to 1)
        assert torch.allclose(y.abs(), torch.ones_like(y.abs()), atol=1e-10)

    def test_power_of_two_size(self):
        """FFT should work optimally with power-of-two sizes."""
        for n in [2, 4, 8, 16, 32, 64, 128]:
            x = torch.randn(n, dtype=torch.float64)
            y = fourier_transform(x)
            assert y.shape == torch.Size([n])

    def test_non_power_of_two_size(self):
        """FFT should work with non-power-of-two sizes."""
        for n in [3, 5, 7, 11, 13, 17, 19, 23]:
            x = torch.randn(n, dtype=torch.float64)
            y = fourier_transform(x)
            assert y.shape == torch.Size([n])
