"""Tests for hilbert_transform and inverse_hilbert_transform."""

import numpy as np
import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import hilbert_transform, inverse_hilbert_transform

# Check if scipy is available for reference tests
try:
    from scipy.signal import hilbert as scipy_hilbert

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestHilbertTransformBasic:
    """Tests for basic hilbert_transform functionality."""

    def test_basic(self):
        """Test basic Hilbert transform."""
        x = torch.randn(32)
        H = hilbert_transform(x)
        assert H.shape == torch.Size([32])
        assert H.dtype == x.dtype  # Real output for real input

    def test_output_is_real_for_real_input(self):
        """Hilbert transform of real input should produce real output."""
        x = torch.randn(32)
        H = hilbert_transform(x)
        assert not H.is_complex()
        assert H.dtype == x.dtype

    def test_output_is_complex_for_complex_input(self):
        """Hilbert transform of complex input should produce complex output."""
        x = torch.randn(32, dtype=torch.complex64)
        H = hilbert_transform(x)
        assert H.is_complex()
        assert H.dtype == x.dtype

    def test_shape_preserved(self):
        """Output shape matches input shape along transform dim."""
        x = torch.randn(10, 20, 30)
        H = hilbert_transform(x, dim=1)
        assert H.shape == x.shape

    def test_with_n_larger(self):
        """Test padding when n > input size."""
        x = torch.randn(64)
        H = hilbert_transform(x, n=128)
        assert H.shape == torch.Size([128])

    def test_with_n_smaller(self):
        """Test truncation when n < input size."""
        x = torch.randn(128)
        H = hilbert_transform(x, n=64)
        assert H.shape == torch.Size([64])

    def test_finite_output(self):
        """Test that output is finite."""
        x = torch.randn(64)
        H = hilbert_transform(x)
        assert torch.all(torch.isfinite(H))


class TestHilbertTransformProperties:
    """Tests for mathematical properties of Hilbert transform."""

    def test_sine_to_cosine(self):
        """Test H[sin(t)] = -cos(t) for positive frequencies."""
        n = 256
        t = torch.linspace(0, 2 * torch.pi, n, dtype=torch.float64)[:-1]
        x = torch.sin(t)
        H = hilbert_transform(x)
        expected = -torch.cos(t)
        # Allow for edge effects
        assert torch.allclose(H[10:-10], expected[10:-10], atol=1e-2)

    def test_cosine_to_sine(self):
        """Test H[cos(t)] = sin(t) for positive frequencies."""
        n = 256
        t = torch.linspace(0, 2 * torch.pi, n, dtype=torch.float64)[:-1]
        x = torch.cos(t)
        H = hilbert_transform(x)
        expected = torch.sin(t)
        # Allow for edge effects
        assert torch.allclose(H[10:-10], expected[10:-10], atol=1e-2)

    def test_double_hilbert_negates(self):
        """Test H[H[f]] = -f (involutory property)."""
        # Use periodic signal for exact property
        n = 256
        t = torch.linspace(0, 2 * torch.pi, n + 1, dtype=torch.float64)[:-1]
        x = torch.sin(t)

        H1 = hilbert_transform(x)
        H2 = hilbert_transform(H1)

        assert torch.allclose(H2, -x, atol=1e-6)

    def test_linearity(self):
        """Test linearity: H[ax + by] = a*H[x] + b*H[y]."""
        x = torch.randn(64, dtype=torch.float64)
        y = torch.randn(64, dtype=torch.float64)
        a, b = 2.5, -1.3

        H_combined = hilbert_transform(a * x + b * y)
        H_separate = a * hilbert_transform(x) + b * hilbert_transform(y)

        assert torch.allclose(H_combined, H_separate, atol=1e-10)

    def test_zero_signal(self):
        """Test that zero input produces zero output."""
        x = torch.zeros(64, dtype=torch.float64)
        H = hilbert_transform(x)
        assert torch.allclose(H, x)


class TestHilbertInverse:
    """Tests for Hilbert / inverse Hilbert relationship."""

    def test_inverse_undoes_forward_periodic(self):
        """Test H^{-1}[H[f]] = f for periodic signal (exact)."""
        # Use a periodic signal to avoid DC component issues
        n = 256
        t = torch.linspace(0, 2 * torch.pi, n + 1, dtype=torch.float64)[:-1]
        x = torch.sin(t) + 0.5 * torch.cos(2 * t)

        H = hilbert_transform(x)
        recovered = inverse_hilbert_transform(H)
        assert torch.allclose(recovered, x, atol=1e-10)

    def test_forward_undoes_inverse_periodic(self):
        """Test H[H^{-1}[f]] = f for periodic signal (exact)."""
        n = 256
        t = torch.linspace(0, 2 * torch.pi, n + 1, dtype=torch.float64)[:-1]
        x = torch.sin(t) + 0.5 * torch.cos(2 * t)

        H_inv = inverse_hilbert_transform(x)
        recovered = hilbert_transform(H_inv)
        assert torch.allclose(recovered, x, atol=1e-10)

    def test_inverse_equals_negative_forward(self):
        """Test H^{-1}[f] = -H[f]."""
        x = torch.randn(64, dtype=torch.float64)
        H = hilbert_transform(x)
        H_inv = inverse_hilbert_transform(x)
        assert torch.allclose(H_inv, -H, atol=1e-10)

    def test_inverse_batched_periodic(self):
        """Test inverse works with batched periodic signal."""
        n = 256
        t = torch.linspace(0, 2 * torch.pi, n + 1, dtype=torch.float64)[:-1]
        # Create batch of periodic signals with different frequencies
        x = torch.stack(
            [
                torch.sin(t),
                torch.cos(t),
                torch.sin(2 * t),
                torch.cos(2 * t),
            ]
        )

        H = hilbert_transform(x, dim=-1)
        recovered = inverse_hilbert_transform(H, dim=-1)
        assert torch.allclose(recovered, x, atol=1e-10)

    def test_inverse_equals_negative_hilbert(self):
        """Test that inverse equals negative forward for any signal."""
        x = torch.randn(64, dtype=torch.float64)
        H_inv = inverse_hilbert_transform(x)
        neg_H = -hilbert_transform(x)
        # This relationship is exact for all signals
        assert torch.allclose(H_inv, neg_H, atol=1e-14)


class TestHilbertTransformMultiDim:
    """Tests for multi-dimensional Hilbert transform."""

    def test_2d(self):
        """Test 2D Hilbert transform."""
        x = torch.randn(16, 32)
        H = hilbert_transform(x, dim=(-2, -1))
        assert H.shape == torch.Size([16, 32])
        assert not H.is_complex()

    def test_2d_double_hilbert(self):
        """Test 2D double Hilbert property for periodic signal."""
        # Create a 2D periodic signal
        n1, n2 = 32, 32
        t1 = torch.linspace(0, 2 * torch.pi, n1 + 1, dtype=torch.float64)[:-1]
        t2 = torch.linspace(0, 2 * torch.pi, n2 + 1, dtype=torch.float64)[:-1]
        x = torch.sin(t1).unsqueeze(1) * torch.cos(t2).unsqueeze(0)

        H1 = hilbert_transform(x, dim=(-2, -1))
        H2 = hilbert_transform(H1, dim=(-2, -1))

        # H[H[f]] = -f along each dimension, so 2D gives f (double negative)
        assert torch.allclose(H2, x, atol=1e-5)

    def test_nd_transform(self):
        """Test nD Hilbert transform."""
        x = torch.randn(4, 8, 16)
        H = hilbert_transform(x, dim=(0, 1, 2))
        assert H.shape == x.shape

    def test_n_with_multi_dim(self):
        """Test n parameter with multi-dim."""
        x = torch.randn(16, 16)
        H = hilbert_transform(x, dim=(-2, -1), n=(32, 32))
        assert H.shape == torch.Size([32, 32])

    def test_n_tuple_truncation(self):
        """Test n parameter for truncation with multi-dim."""
        x = torch.randn(32, 32)
        H = hilbert_transform(x, dim=(-2, -1), n=(16, 16))
        assert H.shape == torch.Size([16, 16])

    def test_single_dim_as_tuple(self):
        """Test single dim provided as tuple."""
        x = torch.randn(32, 64)
        H_int = hilbert_transform(x, dim=-1)
        H_tuple = hilbert_transform(x, dim=(-1,))
        assert torch.allclose(H_int, H_tuple)

    def test_n_tuple_length_mismatch_raises(self):
        """Test that mismatched n and dim tuple lengths raise error."""
        x = torch.randn(16, 16)
        with pytest.raises(ValueError, match="length"):
            hilbert_transform(x, dim=(-2, -1), n=(32,))

    def test_2d_with_batch_dim(self):
        """Test 2D Hilbert preserves batch dimensions."""
        x = torch.randn(4, 8, 16, 32)
        H = hilbert_transform(x, dim=(-2, -1))
        assert H.shape == x.shape


class TestHilbertTransformPadding:
    """Tests for padding modes."""

    def test_constant_padding(self):
        """Test constant (zero) padding."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="constant")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_reflect_padding(self):
        """Test reflect padding."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="reflect")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_replicate_padding(self):
        """Test replicate padding."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="replicate")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_circular_padding(self):
        """Test circular padding."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="circular")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_invalid_padding_mode(self):
        """Test that invalid padding mode raises error."""
        x = torch.randn(32)
        with pytest.raises(ValueError, match="padding_mode"):
            hilbert_transform(x, n=64, padding_mode="invalid")

    def test_linear_padding(self):
        """Test linear extrapolation padding."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="linear")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_smooth_padding(self):
        """Test smooth padding mode."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="smooth")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_polynomial_padding(self):
        """Test polynomial padding mode."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="polynomial")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_spline_padding(self):
        """Test spline padding mode."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="spline")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_padding_order_for_polynomial(self):
        """Test padding_order parameter for polynomial mode."""
        x = torch.randn(32)
        H = hilbert_transform(
            x, n=64, padding_mode="polynomial", padding_order=2
        )
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_reflect_odd_padding(self):
        """Test reflect_odd (antisymmetric) padding mode."""
        x = torch.randn(32)
        H = hilbert_transform(x, n=64, padding_mode="reflect_odd")
        assert H.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H))

    def test_different_padding_modes_differ(self):
        """Test that different padding modes give different results."""
        x = torch.randn(32, dtype=torch.float64)

        H_constant = hilbert_transform(x, n=64, padding_mode="constant")
        H_reflect = hilbert_transform(x, n=64, padding_mode="reflect")

        assert not torch.allclose(H_constant, H_reflect)

    def test_padding_mode_batched(self):
        """Test padding modes work with batched input."""
        x = torch.randn(3, 4, 64, dtype=torch.float64)

        for mode in ["constant", "reflect", "replicate", "circular"]:
            H = hilbert_transform(x, n=128, dim=-1, padding_mode=mode)
            assert H.shape == (3, 4, 128)
            assert torch.all(torch.isfinite(H))


class TestHilbertTransformExplicitPadding:
    """Tests for explicit padding parameter."""

    def test_explicit_padding_1d(self):
        """Test explicit padding for 1D."""
        x = torch.randn(32)
        # Pad 8 on left, 8 on right
        H = hilbert_transform(x, padding=(8, 8))
        assert H.shape == torch.Size([48])

    def test_explicit_padding_asymmetric(self):
        """Test asymmetric explicit padding."""
        x = torch.randn(32)
        H = hilbert_transform(x, padding=(4, 12))
        assert H.shape == torch.Size([48])

    def test_explicit_padding_multi_dim(self):
        """Test explicit padding for multi-dim transform."""
        x = torch.randn(16, 16)
        # Pad each dim by (4, 4)
        H = hilbert_transform(x, dim=(-2, -1), padding=((4, 4), (4, 4)))
        assert H.shape == torch.Size([24, 24])

    def test_n_overrides_padding(self):
        """Test that n parameter works with explicit padding."""
        x = torch.randn(32)
        # Explicit padding of 8 each side would give 48, but n=64 should add more
        H = hilbert_transform(x, padding=(8, 8), n=64)
        assert H.shape == torch.Size([64])


class TestHilbertTransformWindow:
    """Tests for windowing."""

    def test_with_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(32)
        window = torch.hann_window(32)
        H = hilbert_transform(x, window=window)
        assert H.shape == torch.Size([32])
        assert torch.all(torch.isfinite(H))

    def test_with_hamming_window(self):
        """Test with Hamming window."""
        x = torch.randn(32)
        window = torch.hamming_window(32)
        H = hilbert_transform(x, window=window)
        assert H.shape == torch.Size([32])

    def test_rectangular_window_no_effect(self):
        """Test rectangular window (all ones) has no effect."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.ones(64, dtype=torch.float64)

        H_no_window = hilbert_transform(x)
        H_with_window = hilbert_transform(x, window=window)

        assert torch.allclose(H_no_window, H_with_window)

    def test_window_size_mismatch_raises(self):
        """Test that window size mismatch raises error."""
        x = torch.randn(32)
        window = torch.hann_window(64)
        with pytest.raises(ValueError, match="window size"):
            hilbert_transform(x, window=window)

    def test_window_not_1d_raises(self):
        """Test that non-1D window raises error."""
        x = torch.randn(32)
        window = torch.randn(8, 4)
        with pytest.raises(ValueError, match="1-D"):
            hilbert_transform(x, window=window)

    def test_window_multi_dim_raises(self):
        """Test that windowing with multi-dim raises error."""
        x = torch.randn(16, 32)
        window = torch.hann_window(16)
        with pytest.raises(ValueError, match="single-dimension"):
            hilbert_transform(x, dim=(-2, -1), window=window)

    def test_window_with_padding(self):
        """Test window combined with padding."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.hann_window(128, dtype=torch.float64)

        H = hilbert_transform(x, n=128, padding_mode="reflect", window=window)
        assert H.shape == (128,)
        assert torch.all(torch.isfinite(H))

    def test_window_batched(self):
        """Test window broadcasts over batch dimensions."""
        x = torch.randn(3, 4, 64, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        H = hilbert_transform(x, dim=-1, window=window)
        assert H.shape == (3, 4, 64)
        assert torch.all(torch.isfinite(H))

    def test_window_affects_result(self):
        """Test that applying a non-trivial window changes the result."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        H_no_window = hilbert_transform(x)
        H_with_window = hilbert_transform(x, window=window)

        assert not torch.allclose(H_no_window, H_with_window)


class TestHilbertTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_real_input(self):
        """Test gradient correctness for real input."""
        x = torch.randn(16, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            hilbert_transform, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            hilbert_transform, (x,), eps=1e-6, atol=1e-3, rtol=1e-2
        )

    def test_gradcheck_batched(self):
        """Test gradient correctness for batched input."""
        x = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            hilbert_transform, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_multi_dim(self):
        """Test gradient correctness for multi-dim transform."""
        x = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        def func(t):
            return hilbert_transform(t, dim=(-2, -1))

        assert gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradient_with_padding(self):
        """Test gradient works with padding."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return hilbert_transform(
                input_tensor, n=64, padding_mode="reflect"
            )

        assert gradcheck(fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4)

    def test_gradient_with_window(self):
        """Test gradient works with window."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)
        window = torch.hann_window(32, dtype=torch.float64)

        def fn(input_tensor):
            return hilbert_transform(input_tensor, window=window)

        assert gradcheck(fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "mode", ["constant", "reflect", "replicate", "circular"]
    )
    def test_gradcheck_all_padding_modes(self, mode):
        """Test gradient correctness for all padding modes."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return hilbert_transform(input_tensor, n=64, padding_mode=mode)

        assert gradcheck(fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4)


class TestHilbertTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(32, device="cuda")
        H = hilbert_transform(x)
        assert H.device == x.device
        assert torch.all(torch.isfinite(H))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_inverse(self):
        """Test inverse on CUDA."""
        x = torch.randn(32, device="cuda", dtype=torch.float64)
        H = hilbert_transform(x)
        recovered = inverse_hilbert_transform(H)
        assert torch.allclose(recovered, x, atol=1e-5)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_window_device_mismatch_raises(self):
        """Test that window on different device raises error."""
        x = torch.randn(64, device="cuda", dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64, device="cpu")

        with pytest.raises(RuntimeError, match="same device"):
            hilbert_transform(x, window=window)


class TestHilbertTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.randn(32, device="meta")
        H = hilbert_transform(x)

        assert H.shape == torch.Size([32])
        assert H.device == torch.device("meta")

    def test_meta_tensor_with_n(self):
        """Test shape inference with n parameter."""
        x = torch.randn(32, device="meta")
        H = hilbert_transform(x, n=64)

        assert H.shape == torch.Size([64])

    def test_meta_tensor_multi_dim(self):
        """Test shape inference for multi-dim."""
        x = torch.randn(16, 32, device="meta")
        H = hilbert_transform(x, dim=(-2, -1))

        assert H.shape == torch.Size([16, 32])


class TestHilbertTransformDtype:
    """Tests for dtype handling."""

    def test_float32(self):
        """Test float32 input."""
        x = torch.randn(32, dtype=torch.float32)
        H = hilbert_transform(x)
        assert H.dtype == torch.float32

    def test_float64(self):
        """Test float64 input."""
        x = torch.randn(32, dtype=torch.float64)
        H = hilbert_transform(x)
        assert H.dtype == torch.float64

    def test_complex64(self):
        """Test complex64 input."""
        x = torch.randn(32, dtype=torch.complex64)
        H = hilbert_transform(x)
        assert H.dtype == torch.complex64

    def test_complex128(self):
        """Test complex128 input."""
        x = torch.randn(32, dtype=torch.complex128)
        H = hilbert_transform(x)
        assert H.dtype == torch.complex128


class TestHilbertTransformParameterOrder:
    """Tests for new parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        x = torch.randn(32)
        # This should work
        H = hilbert_transform(x, dim=-1, n=64)
        assert H.shape == torch.Size([64])

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        x = torch.randn(32)
        # This should fail - dim should be keyword only
        with pytest.raises(TypeError):
            hilbert_transform(x, -1)  # type: ignore


class TestHilbertTransformOut:
    """Tests for out parameter."""

    def test_out_parameter(self):
        """Test that out parameter works."""
        x = torch.randn(32, dtype=torch.float64)
        out = torch.empty(32, dtype=torch.float64)
        result = hilbert_transform(x, out=out)

        assert result is out
        # Verify correctness
        expected = hilbert_transform(x)
        assert torch.allclose(out, expected)

    def test_out_parameter_with_n(self):
        """Test out parameter with n."""
        x = torch.randn(32)
        out = torch.empty(64)
        result = hilbert_transform(x, n=64, out=out)

        assert result is out
        assert out.shape == torch.Size([64])


class TestHilbertTransformComplex:
    """Tests for complex input."""

    def test_complex_input_basic(self):
        """Test complex input basic functionality."""
        x = torch.randn(64, dtype=torch.complex128)
        H = hilbert_transform(x)
        assert H.shape == (64,)
        assert H.dtype == torch.complex128
        assert torch.all(torch.isfinite(H.real))
        assert torch.all(torch.isfinite(H.imag))

    @pytest.mark.parametrize(
        "mode", ["constant", "reflect", "replicate", "circular"]
    )
    def test_complex_input_with_padding_modes(self, mode):
        """Test complex input works with all padding modes."""
        x = torch.randn(64, dtype=torch.complex128)
        H = hilbert_transform(x, n=128, padding_mode=mode)
        assert H.shape == (128,)
        assert H.dtype == torch.complex128

    def test_complex_input_batched_with_padding(self):
        """Test complex batched input with padding."""
        x = torch.randn(3, 4, 64, dtype=torch.complex128)
        H = hilbert_transform(x, n=128, dim=-1, padding_mode="reflect")
        assert H.shape == (3, 4, 128)
        assert H.dtype == torch.complex128


class TestHilbertTransformTruncation:
    """Tests for truncation mode (n < input_size)."""

    def test_truncation_basic(self):
        """Test basic truncation."""
        x = torch.randn(128, dtype=torch.float64)
        H = hilbert_transform(x, n=64)
        assert H.shape == (64,)
        assert torch.all(torch.isfinite(H))

    def test_truncation_gradient(self):
        """Test gradient with truncation."""
        x = torch.randn(128, requires_grad=True, dtype=torch.float64)
        H = hilbert_transform(x, n=64)
        loss = H.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_gradcheck_truncation(self):
        """Test gradient correctness with truncation."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return hilbert_transform(input_tensor, n=32)

        assert gradcheck(fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4)

    def test_truncation_batched(self):
        """Test truncation with batched input."""
        x = torch.randn(3, 4, 128, dtype=torch.float64)
        H = hilbert_transform(x, n=64, dim=-1)
        assert H.shape == (3, 4, 64)


class TestHilbertTransformNonLastDim:
    """Tests for transform along non-last dimension."""

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_3d_all_dims(self, dim):
        """Test 3D input with transform along all possible dimensions."""
        x = torch.randn(16, 32, 64, dtype=torch.float64)
        H = hilbert_transform(x, dim=dim)
        assert H.shape == x.shape
        assert torch.all(torch.isfinite(H))

    @pytest.mark.parametrize("dim", [0, 1])
    def test_3d_non_last_dim_with_padding(self, dim):
        """Test padding along non-last dimension."""
        x = torch.randn(16, 32, 64, dtype=torch.float64)
        n = x.size(dim) * 2
        H = hilbert_transform(x, n=n, dim=dim, padding_mode="reflect")

        expected_shape = list(x.shape)
        expected_shape[dim] = n
        assert H.shape == tuple(expected_shape)
        assert torch.all(torch.isfinite(H))

    def test_dim_0_gradient(self):
        """Test gradient with dim=0."""
        x = torch.randn(32, 64, requires_grad=True, dtype=torch.float64)
        H = hilbert_transform(x, dim=0)
        loss = H.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_negative_dim(self):
        """Test negative dimension indexing."""
        x = torch.randn(16, 32, 64, dtype=torch.float64)

        H_neg = hilbert_transform(x, dim=-2)
        H_pos = hilbert_transform(x, dim=1)

        assert torch.allclose(H_neg, H_pos)


class TestInverseHilbertTransform:
    """Tests specific to inverse_hilbert_transform."""

    def test_basic(self):
        """Test basic inverse Hilbert transform."""
        x = torch.randn(32)
        H_inv = inverse_hilbert_transform(x)
        assert H_inv.shape == torch.Size([32])
        assert H_inv.dtype == x.dtype

    def test_multi_dim(self):
        """Test multi-dim inverse Hilbert transform."""
        x = torch.randn(16, 32)
        H_inv = inverse_hilbert_transform(x, dim=(-2, -1))
        assert H_inv.shape == x.shape

    def test_with_n(self):
        """Test inverse Hilbert transform with n parameter."""
        x = torch.randn(32)
        H_inv = inverse_hilbert_transform(x, n=64)
        assert H_inv.shape == torch.Size([64])

    def test_with_padding(self):
        """Test inverse Hilbert transform with padding."""
        x = torch.randn(32)
        H_inv = inverse_hilbert_transform(x, n=64, padding_mode="reflect")
        assert H_inv.shape == torch.Size([64])
        assert torch.all(torch.isfinite(H_inv))

    def test_with_window(self):
        """Test inverse Hilbert transform with window."""
        x = torch.randn(32)
        window = torch.hann_window(32)
        H_inv = inverse_hilbert_transform(x, window=window)
        assert H_inv.shape == torch.Size([32])

    def test_gradcheck(self):
        """Test gradient correctness for inverse Hilbert transform."""
        x = torch.randn(16, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            inverse_hilbert_transform, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )


@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
class TestHilbertTransformScipyReference:
    """Tests comparing against SciPy's hilbert implementation."""

    def test_matches_scipy_basic(self):
        """Test basic case matches SciPy."""
        np.random.seed(42)
        x_np = np.random.randn(128)
        x_torch = torch.from_numpy(x_np)

        # SciPy's hilbert returns analytic signal: f + i*H[f]
        # So the imaginary part is the Hilbert transform
        scipy_result = scipy_hilbert(x_np).imag

        torch_result = hilbert_transform(x_torch).numpy()

        np.testing.assert_allclose(
            scipy_result, torch_result, rtol=1e-10, atol=1e-10
        )

    def test_matches_scipy_various_sizes(self):
        """Test various signal sizes match SciPy."""
        np.random.seed(42)
        for n in [32, 64, 100, 127, 128, 255, 256]:
            x_np = np.random.randn(n)
            x_torch = torch.from_numpy(x_np)

            scipy_result = scipy_hilbert(x_np).imag
            torch_result = hilbert_transform(x_torch).numpy()

            np.testing.assert_allclose(
                scipy_result,
                torch_result,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Mismatch for n={n}",
            )

    def test_matches_scipy_sine_wave(self):
        """Test with sine wave (known analytical result)."""
        n = 256
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_np = np.sin(t)
        x_torch = torch.from_numpy(x_np)

        scipy_result = scipy_hilbert(x_np).imag
        torch_result = hilbert_transform(x_torch).numpy()

        np.testing.assert_allclose(
            scipy_result, torch_result, rtol=1e-10, atol=1e-10
        )

        # For sin(t), H[sin(t)] = -cos(t)
        expected = -np.cos(t)
        np.testing.assert_allclose(
            torch_result, expected, rtol=1e-4, atol=1e-4
        )

    def test_matches_scipy_cosine_wave(self):
        """Test with cosine wave (known analytical result)."""
        n = 256
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_np = np.cos(t)
        x_torch = torch.from_numpy(x_np)

        scipy_result = scipy_hilbert(x_np).imag
        torch_result = hilbert_transform(x_torch).numpy()

        np.testing.assert_allclose(
            scipy_result, torch_result, rtol=1e-10, atol=1e-10
        )

        # For cos(t), H[cos(t)] = sin(t)
        expected = np.sin(t)
        np.testing.assert_allclose(
            torch_result, expected, rtol=1e-4, atol=1e-4
        )

    def test_matches_scipy_with_n_parameter(self):
        """Test with n parameter (padding) matches SciPy."""
        np.random.seed(42)
        x_np = np.random.randn(64)
        x_torch = torch.from_numpy(x_np)

        # With n > input_size, both should zero-pad
        scipy_result = scipy_hilbert(x_np, N=128).imag
        torch_result = hilbert_transform(
            x_torch, n=128, padding_mode="constant", padding_value=0.0
        ).numpy()

        np.testing.assert_allclose(
            scipy_result, torch_result, rtol=1e-10, atol=1e-10
        )
