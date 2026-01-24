"""Tests for sine_transform and inverse_sine_transform."""

import numpy as np
import pytest
import torch
from scipy import fft as scipy_fft
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import (
    fourier_sine_transform,
    inverse_fourier_sine_transform,
    inverse_sine_transform,
    sine_transform,
)


class TestSineTransformBasic:
    """Basic functionality tests for sine_transform."""

    def test_basic_dst2(self):
        """Test basic DST-II (default)."""
        x = torch.randn(32)
        X = sine_transform(x)
        assert X.shape == torch.Size([32])

    def test_dst_types(self):
        """Test all four DST types."""
        x = torch.randn(32)
        for t in [1, 2, 3, 4]:
            X = sine_transform(x, type=t)
            assert X.shape == torch.Size([32])

    def test_complex_input_raises(self):
        """DST should raise error for complex input."""
        x = torch.randn(32, dtype=torch.complex64)
        with pytest.raises(ValueError, match="real"):
            sine_transform(x)

    def test_invalid_type_raises(self):
        """Invalid DST type should raise error."""
        x = torch.randn(32)
        with pytest.raises(ValueError, match="type must be"):
            sine_transform(x, type=5)

    def test_output_is_real(self):
        """DST output should be real-valued."""
        x = torch.randn(16, dtype=torch.float64)
        for t in [1, 2, 3, 4]:
            y = sine_transform(x, type=t)
            assert y.dtype == torch.float64
            assert not y.is_complex()

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = torch.randn(32, dtype=torch.float64)
        y = sine_transform(x, type=2)
        assert y.shape == x.shape

    def test_batched_input(self):
        """DST should work with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        y = sine_transform(x, type=2, dim=-1)
        assert y.shape == x.shape

    def test_different_dims(self):
        """DST should work along different dimensions."""
        x = torch.randn(4, 8, 16, dtype=torch.float64)
        for dim in [0, 1, 2, -1, -2, -3]:
            y = sine_transform(x, type=2, dim=dim)
            assert y.shape == x.shape


class TestSineTransformScipyComparison:
    """Test DST against scipy reference implementation."""

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_matches_scipy_backward_norm(self, dst_type):
        """DST should match scipy with backward normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = sine_transform(x, type=dst_type, norm="backward")
        scipy_val = scipy_fft.dst(x.numpy(), type=dst_type, norm=None)
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_matches_scipy_ortho_norm(self, dst_type):
        """DST should match scipy with ortho normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = sine_transform(x, type=dst_type, norm="ortho")
        scipy_val = scipy_fft.dst(x.numpy(), type=dst_type, norm="ortho")
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)


class TestSineTransformMultiDim:
    """Tests for multi-dimensional DST."""

    def test_2d_dst(self):
        """Test 2D DST with dim tuple."""
        x = torch.randn(16, 32)
        X = sine_transform(x, dim=(-2, -1))
        assert X.shape == torch.Size([16, 32])

    def test_2d_with_n(self):
        """Test 2D DST with n parameter."""
        x = torch.randn(16, 16)
        X = sine_transform(x, dim=(-2, -1), n=(32, 32))
        assert X.shape == torch.Size([32, 32])

    def test_3d_dst(self):
        """Test 3D DST."""
        x = torch.randn(8, 16, 32)
        X = sine_transform(x, dim=(0, 1, 2))
        assert X.shape == x.shape

    def test_2d_with_batch(self):
        """Test 2D DST preserves batch dimensions."""
        x = torch.randn(4, 8, 16, 32)
        X = sine_transform(x, dim=(-2, -1))
        assert X.shape == x.shape

    def test_single_dim_as_tuple(self):
        """Test single dim provided as tuple."""
        x = torch.randn(32, 64)
        X1 = sine_transform(x, dim=-1)
        X2 = sine_transform(x, dim=(-1,))
        assert torch.allclose(X1, X2)

    def test_n_tuple_length_mismatch_raises(self):
        """Test that mismatched n and dim tuple lengths raise error."""
        x = torch.randn(16, 16)
        with pytest.raises(ValueError, match="length"):
            sine_transform(x, dim=(-2, -1), n=(32,))


class TestSineTransformRoundTrip:
    """Test DST -> IDST round-trip."""

    def test_round_trip_type2_ortho(self):
        """Round-trip DST-II with ortho normalization."""
        x = torch.randn(32)
        X = sine_transform(x, type=2, norm="ortho")
        x_rec = inverse_sine_transform(X, type=2, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_round_trip_type1(self):
        """Round-trip DST-I (self-inverse up to scaling)."""
        x = torch.randn(32)
        X = sine_transform(x, type=1, norm="ortho")
        x_rec = inverse_sine_transform(X, type=1, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_round_trip_type4(self):
        """Round-trip DST-IV (self-inverse)."""
        x = torch.randn(32)
        X = sine_transform(x, type=4, norm="ortho")
        x_rec = inverse_sine_transform(X, type=4, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-4)

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_round_trip_all_types(self, dst_type, norm):
        """Round-trip for all DST types and normalizations."""
        x = torch.randn(16, dtype=torch.float64)
        X = sine_transform(x, type=dst_type, norm=norm)
        x_rec = inverse_sine_transform(X, type=dst_type, norm=norm)
        assert torch.allclose(x_rec, x, atol=1e-10)

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    def test_round_trip_batched(self, dst_type):
        """Round-trip with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        X = sine_transform(x, type=dst_type, norm="ortho", dim=-1)
        x_rec = inverse_sine_transform(X, type=dst_type, norm="ortho", dim=-1)
        assert torch.allclose(x_rec, x, atol=1e-10)

    def test_round_trip_2d(self):
        """Round-trip 2D DST."""
        x = torch.randn(16, 32, dtype=torch.float64)
        X = sine_transform(x, dim=(-2, -1), norm="ortho")
        x_rec = inverse_sine_transform(X, dim=(-2, -1), norm="ortho")
        assert torch.allclose(x_rec, x, atol=1e-10)


class TestSineTransformWithN:
    """Test DST with signal length parameter."""

    def test_n_larger_pads(self):
        """When n > input_size, input should be zero-padded."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = sine_transform(x, n=8, type=2)
        assert y.shape == torch.Size([8])

    def test_n_smaller_truncates(self):
        """When n < input_size, input should be truncated."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        y = sine_transform(x, n=4, type=2)
        assert y.shape == torch.Size([4])

    def test_n_matches_input(self):
        """When n == input_size, no change."""
        x = torch.randn(32)
        y = sine_transform(x, n=32)
        # Should be identical to no-n version
        y_no_n = sine_transform(x)
        assert torch.allclose(y, y_no_n)


class TestSineTransformPadding:
    """Tests for padding modes."""

    def test_constant_padding(self):
        """Test constant (zero) padding."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="constant")
        assert X.shape == torch.Size([64])

    def test_reflect_padding(self):
        """Test reflect padding."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="reflect")
        assert X.shape == torch.Size([64])

    def test_replicate_padding(self):
        """Test replicate padding."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="replicate")
        assert X.shape == torch.Size([64])

    def test_circular_padding(self):
        """Test circular padding."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="circular")
        assert X.shape == torch.Size([64])

    def test_linear_padding(self):
        """Test linear extrapolation padding."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="linear")
        assert X.shape == torch.Size([64])

    def test_smooth_padding(self):
        """Test smooth padding mode."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="smooth")
        assert X.shape == torch.Size([64])

    def test_polynomial_padding(self):
        """Test polynomial padding mode."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="polynomial")
        assert X.shape == torch.Size([64])

    def test_spline_padding(self):
        """Test spline padding mode."""
        x = torch.randn(32)
        X = sine_transform(x, n=64, padding_mode="spline")
        assert X.shape == torch.Size([64])

    def test_invalid_padding_mode(self):
        """Test that invalid padding mode raises error."""
        x = torch.randn(32)
        with pytest.raises(ValueError, match="padding_mode"):
            sine_transform(x, n=64, padding_mode="invalid")


class TestSineTransformExplicitPadding:
    """Tests for explicit padding parameter."""

    def test_explicit_padding_1d(self):
        """Test explicit padding for 1D."""
        x = torch.randn(32)
        # Pad 8 on left, 8 on right
        X = sine_transform(x, padding=(8, 8))
        assert X.shape == torch.Size([48])

    def test_explicit_padding_asymmetric(self):
        """Test asymmetric explicit padding."""
        x = torch.randn(32)
        X = sine_transform(x, padding=(4, 12))
        assert X.shape == torch.Size([48])

    def test_explicit_padding_multi_dim(self):
        """Test explicit padding for multi-dim transform."""
        x = torch.randn(16, 16)
        # Pad each dim by (4, 4)
        X = sine_transform(x, dim=(-2, -1), padding=((4, 4), (4, 4)))
        assert X.shape == torch.Size([24, 24])

    def test_n_overrides_padding(self):
        """Test that n parameter works with explicit padding."""
        x = torch.randn(32)
        # Explicit padding would give 48, but n=64 should extend further
        X = sine_transform(x, padding=(8, 8), n=64)
        assert X.shape == torch.Size([64])


class TestSineTransformWindow:
    """Tests for windowing."""

    def test_with_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(32)
        window = torch.hann_window(32)
        X = sine_transform(x, window=window)

        # Compare to manual windowing
        expected = sine_transform(x * window)
        assert torch.allclose(X, expected)

    def test_with_hamming_window(self):
        """Test with Hamming window."""
        x = torch.randn(32)
        window = torch.hamming_window(32)
        X = sine_transform(x, window=window)

        expected = sine_transform(x * window)
        assert torch.allclose(X, expected)

    def test_window_size_mismatch_raises(self):
        """Window size mismatch should raise error."""
        x = torch.randn(32)
        window = torch.hann_window(64)  # Wrong size
        with pytest.raises(ValueError, match="window size"):
            sine_transform(x, window=window)

    def test_window_multi_dim_raises(self):
        """Windowing with multi-dim transform should raise error."""
        x = torch.randn(16, 16)
        window = torch.hann_window(16)
        with pytest.raises(ValueError, match="single-dimension"):
            sine_transform(x, dim=(-2, -1), window=window)


class TestSineTransformGradient:
    """Test DST gradient correctness."""

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dst_type, norm):
        """Gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: sine_transform(inp, type=dst_type, norm=norm),
            (x,),
            raise_exception=True,
        )

    @pytest.mark.parametrize("dst_type", [2, 4])
    def test_gradgradcheck(self, dst_type):
        """Second-order gradient should pass numerical check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            lambda inp: sine_transform(inp, type=dst_type, norm="ortho"),
            (x,),
            raise_exception=True,
        )


class TestInverseSineTransformGradient:
    """Test IDST gradient correctness."""

    @pytest.mark.parametrize("dst_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dst_type, norm):
        """IDST gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: inverse_sine_transform(inp, type=dst_type, norm=norm),
            (x,),
            raise_exception=True,
        )


class TestSineTransformMeta:
    """Test DST with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = sine_transform(x, type=2)
        assert y.shape == torch.Size([16])
        assert y.device.type == "meta"

    def test_meta_tensor_with_n(self):
        """Meta tensor with n parameter should produce correct shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = sine_transform(x, n=32, type=2)
        assert y.shape == torch.Size([32])

    def test_meta_tensor_multi_dim(self):
        """Meta tensor with multi-dim DST."""
        x = torch.empty(16, 32, device="meta", dtype=torch.float64)
        y = sine_transform(x, dim=(-2, -1))
        assert y.shape == torch.Size([16, 32])


class TestSineTransformDevice:
    """Test DST on different devices."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """DST should work on CUDA tensors."""
        x = torch.randn(16, dtype=torch.float64, device="cuda")
        y = sine_transform(x, type=2)
        assert y.device.type == "cuda"

        # Round-trip should work
        x_rec = inverse_sine_transform(y, type=2)
        assert torch.allclose(x_rec.cpu(), x.cpu(), atol=1e-10)


class TestSineTransformNormalization:
    """Tests for normalization modes."""

    def test_backward_norm(self):
        """Test backward normalization."""
        x = torch.randn(32, dtype=torch.float64)
        X = sine_transform(x, norm="backward")
        # Compare with scipy
        X_scipy = scipy_fft.dst(x.numpy(), type=2, norm=None)
        assert np.allclose(X.numpy(), X_scipy, atol=1e-10)

    def test_ortho_norm(self):
        """Test ortho normalization."""
        x = torch.randn(32, dtype=torch.float64)
        X = sine_transform(x, norm="ortho")
        # Compare with scipy
        X_scipy = scipy_fft.dst(x.numpy(), type=2, norm="ortho")
        assert np.allclose(X.numpy(), X_scipy, atol=1e-10)

    def test_forward_norm(self):
        """Test forward normalization."""
        x = torch.randn(32, dtype=torch.float64)
        X = sine_transform(x, norm="forward")
        # Compare with scipy
        X_scipy = scipy_fft.dst(x.numpy(), type=2, norm="forward")
        assert np.allclose(X.numpy(), X_scipy, atol=1e-10)


class TestSineTransformBackwardCompatibility:
    """Tests for backward compatibility with fourier_sine_transform."""

    def test_alias_exists(self):
        """fourier_sine_transform should be an alias."""
        assert fourier_sine_transform is sine_transform

    def test_inverse_alias_exists(self):
        """inverse_fourier_sine_transform should be an alias."""
        assert inverse_fourier_sine_transform is inverse_sine_transform

    def test_old_api_works(self):
        """Old API should still work."""
        x = torch.randn(32)
        X = fourier_sine_transform(x, type=2, norm="ortho")
        x_rec = inverse_fourier_sine_transform(X, type=2, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)


class TestSineTransformParameterOrder:
    """Tests for new parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        x = torch.randn(32)
        # This should work
        X = sine_transform(x, type=2, dim=-1, n=64, norm="ortho")
        assert X.shape == torch.Size([64])

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        x = torch.randn(32)
        # This should fail - type should be keyword only
        with pytest.raises(TypeError):
            sine_transform(x, 2)  # type: ignore


class TestInverseSineTransformBasic:
    """Basic functionality tests for inverse_sine_transform."""

    def test_basic_idst2(self):
        """Test basic IDST-II (default)."""
        X = torch.randn(32)
        x = inverse_sine_transform(X)
        assert x.shape == torch.Size([32])

    def test_complex_input_raises(self):
        """IDST should raise error for complex input."""
        X = torch.randn(32, dtype=torch.complex64)
        with pytest.raises(ValueError, match="real"):
            inverse_sine_transform(X)

    def test_output_is_real(self):
        """IDST output should be real-valued."""
        X = torch.randn(16, dtype=torch.float64)
        for t in [1, 2, 3, 4]:
            x = inverse_sine_transform(X, type=t)
            assert x.dtype == torch.float64
            assert not x.is_complex()


class TestInverseSineTransformMultiDim:
    """Tests for multi-dimensional IDST."""

    def test_2d_idst(self):
        """Test 2D IDST with dim tuple."""
        X = torch.randn(16, 32)
        x = inverse_sine_transform(X, dim=(-2, -1))
        assert x.shape == torch.Size([16, 32])

    def test_2d_with_n(self):
        """Test 2D IDST with n parameter."""
        X = torch.randn(16, 16)
        x = inverse_sine_transform(X, dim=(-2, -1), n=(32, 32))
        assert x.shape == torch.Size([32, 32])


class TestInverseSineTransformPadding:
    """Tests for inverse transform padding modes."""

    def test_constant_padding_inverse(self):
        """Test constant padding for inverse."""
        X = torch.randn(32)
        x = inverse_sine_transform(X, n=64, padding_mode="constant")
        assert x.shape == torch.Size([64])

    def test_linear_padding_inverse(self):
        """Test linear padding for inverse."""
        X = torch.randn(32)
        x = inverse_sine_transform(X, n=64, padding_mode="linear")
        assert x.shape == torch.Size([64])

    def test_invalid_padding_mode_inverse(self):
        """Test that invalid padding mode raises error for inverse."""
        X = torch.randn(32)
        with pytest.raises(ValueError, match="padding_mode"):
            inverse_sine_transform(X, n=64, padding_mode="invalid")


class TestInverseSineTransformWindow:
    """Tests for inverse transform windowing."""

    def test_with_hann_window_inverse(self):
        """Test with Hann window for inverse (applied after IDST)."""
        X = torch.randn(32)
        window = torch.hann_window(32)
        x = inverse_sine_transform(X, window=window)

        # Compare to manual windowing (after transform)
        expected = inverse_sine_transform(X) * window
        assert torch.allclose(x, expected)


class TestInverseSineTransformExplicitPadding:
    """Tests for inverse transform explicit padding parameter."""

    def test_explicit_padding_1d_inverse(self):
        """Test explicit padding for 1D inverse."""
        X = torch.randn(32)
        x = inverse_sine_transform(X, padding=(8, 8))
        assert x.shape == torch.Size([48])

    def test_explicit_padding_multi_dim_inverse(self):
        """Test explicit padding for multi-dim inverse."""
        X = torch.randn(16, 16)
        x = inverse_sine_transform(X, dim=(-2, -1), padding=((4, 4), (4, 4)))
        assert x.shape == torch.Size([24, 24])
