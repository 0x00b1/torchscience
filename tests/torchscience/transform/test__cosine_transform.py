"""Tests for cosine_transform and inverse_cosine_transform."""

import numpy as np
import pytest
import torch
from scipy import fft as scipy_fft
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import (
    cosine_transform,
    fourier_cosine_transform,
    inverse_cosine_transform,
    inverse_fourier_cosine_transform,
)


class TestCosineTransformBasic:
    """Basic functionality tests for cosine_transform."""

    def test_basic_dct2(self):
        """Test basic DCT-II (default)."""
        x = torch.randn(32)
        X = cosine_transform(x)
        assert X.shape == torch.Size([32])

    def test_dct_types(self):
        """Test all four DCT types."""
        x = torch.randn(32)
        for t in [1, 2, 3, 4]:
            X = cosine_transform(x, type=t)
            assert X.shape == torch.Size([32])

    def test_complex_input_raises(self):
        """DCT should raise error for complex input."""
        x = torch.randn(32, dtype=torch.complex64)
        with pytest.raises(ValueError, match="real"):
            cosine_transform(x)

    def test_invalid_type_raises(self):
        """Invalid DCT type should raise error."""
        x = torch.randn(32)
        with pytest.raises(ValueError, match="type must be"):
            cosine_transform(x, type=5)

    def test_output_is_real(self):
        """DCT output should be real-valued."""
        x = torch.randn(16, dtype=torch.float64)
        for t in [1, 2, 3, 4]:
            y = cosine_transform(x, type=t)
            assert y.dtype == torch.float64
            assert not y.is_complex()

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        x = torch.randn(32, dtype=torch.float64)
        y = cosine_transform(x, type=2)
        assert y.shape == x.shape

    def test_batched_input(self):
        """DCT should work with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        y = cosine_transform(x, type=2, dim=-1)
        assert y.shape == x.shape

    def test_different_dims(self):
        """DCT should work along different dimensions."""
        x = torch.randn(4, 8, 16, dtype=torch.float64)
        for dim in [0, 1, 2, -1, -2, -3]:
            y = cosine_transform(x, type=2, dim=dim)
            assert y.shape == x.shape


class TestCosineTransformScipyComparison:
    """Test DCT against scipy reference implementation."""

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_matches_scipy_backward_norm(self, dct_type):
        """DCT should match scipy with backward normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = cosine_transform(x, type=dct_type, norm="backward")
        scipy_val = scipy_fft.dct(x.numpy(), type=dct_type, norm=None)
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_matches_scipy_ortho_norm(self, dct_type):
        """DCT should match scipy with ortho normalization."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        our = cosine_transform(x, type=dct_type, norm="ortho")
        scipy_val = scipy_fft.dct(x.numpy(), type=dct_type, norm="ortho")
        assert np.allclose(our.numpy(), scipy_val, atol=1e-10)


class TestCosineTransformMultiDim:
    """Tests for multi-dimensional DCT."""

    def test_2d_dct(self):
        """Test 2D DCT with dim tuple."""
        x = torch.randn(16, 32)
        X = cosine_transform(x, dim=(-2, -1))
        assert X.shape == torch.Size([16, 32])

    def test_2d_with_n(self):
        """Test 2D DCT with n parameter."""
        x = torch.randn(16, 16)
        X = cosine_transform(x, dim=(-2, -1), n=(32, 32))
        assert X.shape == torch.Size([32, 32])

    def test_3d_dct(self):
        """Test 3D DCT."""
        x = torch.randn(8, 16, 32)
        X = cosine_transform(x, dim=(0, 1, 2))
        assert X.shape == x.shape

    def test_2d_with_batch(self):
        """Test 2D DCT preserves batch dimensions."""
        x = torch.randn(4, 8, 16, 32)
        X = cosine_transform(x, dim=(-2, -1))
        assert X.shape == x.shape

    def test_single_dim_as_tuple(self):
        """Test single dim provided as tuple."""
        x = torch.randn(32, 64)
        X1 = cosine_transform(x, dim=-1)
        X2 = cosine_transform(x, dim=(-1,))
        assert torch.allclose(X1, X2)

    def test_n_tuple_length_mismatch_raises(self):
        """Test that mismatched n and dim tuple lengths raise error."""
        x = torch.randn(16, 16)
        with pytest.raises(ValueError, match="length"):
            cosine_transform(x, dim=(-2, -1), n=(32,))


class TestCosineTransformRoundTrip:
    """Test DCT -> IDCT round-trip."""

    def test_round_trip_type2_ortho(self):
        """Round-trip DCT-II with ortho normalization."""
        x = torch.randn(32)
        X = cosine_transform(x, type=2, norm="ortho")
        x_rec = inverse_cosine_transform(X, type=2, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_round_trip_type1(self):
        """Round-trip DCT-I (self-inverse)."""
        x = torch.randn(32)
        X = cosine_transform(x, type=1, norm="ortho")
        x_rec = inverse_cosine_transform(X, type=1, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_round_trip_type4(self):
        """Round-trip DCT-IV (self-inverse)."""
        x = torch.randn(32)
        X = cosine_transform(x, type=4, norm="ortho")
        x_rec = inverse_cosine_transform(X, type=4, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-4)

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_round_trip_all_types(self, dct_type, norm):
        """Round-trip for all DCT types and normalizations."""
        x = torch.randn(16, dtype=torch.float64)
        X = cosine_transform(x, type=dct_type, norm=norm)
        x_rec = inverse_cosine_transform(X, type=dct_type, norm=norm)
        assert torch.allclose(x_rec, x, atol=1e-10)

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_round_trip_batched(self, dct_type):
        """Round-trip with batched inputs."""
        x = torch.randn(3, 5, 16, dtype=torch.float64)
        X = cosine_transform(x, type=dct_type, norm="ortho", dim=-1)
        x_rec = inverse_cosine_transform(
            X, type=dct_type, norm="ortho", dim=-1
        )
        assert torch.allclose(x_rec, x, atol=1e-10)

    def test_round_trip_2d(self):
        """Round-trip 2D DCT."""
        x = torch.randn(16, 32, dtype=torch.float64)
        X = cosine_transform(x, dim=(-2, -1), norm="ortho")
        x_rec = inverse_cosine_transform(X, dim=(-2, -1), norm="ortho")
        assert torch.allclose(x_rec, x, atol=1e-10)


class TestCosineTransformWithN:
    """Test DCT with signal length parameter."""

    def test_n_larger_pads(self):
        """When n > input_size, input should be zero-padded."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        y = cosine_transform(x, n=8, type=2)
        assert y.shape == torch.Size([8])

    def test_n_smaller_truncates(self):
        """When n < input_size, input should be truncated."""
        x = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64
        )
        y = cosine_transform(x, n=4, type=2)
        assert y.shape == torch.Size([4])

    def test_n_matches_input(self):
        """When n == input_size, no change."""
        x = torch.randn(32)
        y = cosine_transform(x, n=32)
        # Should be identical to no-n version
        y_no_n = cosine_transform(x)
        assert torch.allclose(y, y_no_n)


class TestCosineTransformPadding:
    """Tests for padding modes."""

    def test_constant_padding(self):
        """Test constant (zero) padding."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="constant")
        assert X.shape == torch.Size([64])

    def test_reflect_padding(self):
        """Test reflect padding."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="reflect")
        assert X.shape == torch.Size([64])

    def test_replicate_padding(self):
        """Test replicate padding."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="replicate")
        assert X.shape == torch.Size([64])

    def test_circular_padding(self):
        """Test circular padding."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="circular")
        assert X.shape == torch.Size([64])

    def test_linear_padding(self):
        """Test linear extrapolation padding."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="linear")
        assert X.shape == torch.Size([64])

    def test_smooth_padding(self):
        """Test smooth padding mode."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="smooth")
        assert X.shape == torch.Size([64])

    def test_polynomial_padding(self):
        """Test polynomial padding mode."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="polynomial")
        assert X.shape == torch.Size([64])

    def test_spline_padding(self):
        """Test spline padding mode."""
        x = torch.randn(32)
        X = cosine_transform(x, n=64, padding_mode="spline")
        assert X.shape == torch.Size([64])

    def test_invalid_padding_mode(self):
        """Test that invalid padding mode raises error."""
        x = torch.randn(32)
        with pytest.raises(ValueError, match="padding_mode"):
            cosine_transform(x, n=64, padding_mode="invalid")


class TestCosineTransformExplicitPadding:
    """Tests for explicit padding parameter."""

    def test_explicit_padding_1d(self):
        """Test explicit padding for 1D."""
        x = torch.randn(32)
        # Pad 8 on left, 8 on right
        X = cosine_transform(x, padding=(8, 8))
        assert X.shape == torch.Size([48])

    def test_explicit_padding_asymmetric(self):
        """Test asymmetric explicit padding."""
        x = torch.randn(32)
        X = cosine_transform(x, padding=(4, 12))
        assert X.shape == torch.Size([48])

    def test_explicit_padding_multi_dim(self):
        """Test explicit padding for multi-dim transform."""
        x = torch.randn(16, 16)
        # Pad each dim by (4, 4)
        X = cosine_transform(x, dim=(-2, -1), padding=((4, 4), (4, 4)))
        assert X.shape == torch.Size([24, 24])

    def test_n_overrides_padding(self):
        """Test that n parameter works with explicit padding."""
        x = torch.randn(32)
        # Explicit padding would give 48, but n=64 should extend further
        X = cosine_transform(x, padding=(8, 8), n=64)
        assert X.shape == torch.Size([64])


class TestCosineTransformWindow:
    """Tests for windowing."""

    def test_with_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(32)
        window = torch.hann_window(32)
        X = cosine_transform(x, window=window)

        # Compare to manual windowing
        expected = cosine_transform(x * window)
        assert torch.allclose(X, expected)

    def test_with_hamming_window(self):
        """Test with Hamming window."""
        x = torch.randn(32)
        window = torch.hamming_window(32)
        X = cosine_transform(x, window=window)

        expected = cosine_transform(x * window)
        assert torch.allclose(X, expected)

    def test_window_size_mismatch_raises(self):
        """Window size mismatch should raise error."""
        x = torch.randn(32)
        window = torch.hann_window(64)  # Wrong size
        with pytest.raises(ValueError, match="window size"):
            cosine_transform(x, window=window)

    def test_window_multi_dim_raises(self):
        """Windowing with multi-dim transform should raise error."""
        x = torch.randn(16, 16)
        window = torch.hann_window(16)
        with pytest.raises(ValueError, match="single-dimension"):
            cosine_transform(x, dim=(-2, -1), window=window)


class TestCosineTransformGradient:
    """Test DCT gradient correctness."""

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dct_type, norm):
        """Gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: cosine_transform(inp, type=dct_type, norm=norm),
            (x,),
            raise_exception=True,
        )

    @pytest.mark.parametrize("dct_type", [2, 4])
    def test_gradgradcheck(self, dct_type):
        """Second-order gradient should pass numerical check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            lambda inp: cosine_transform(inp, type=dct_type, norm="ortho"),
            (x,),
            raise_exception=True,
        )


class TestInverseCosineTransformGradient:
    """Test IDCT gradient correctness."""

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    @pytest.mark.parametrize("norm", ["backward", "ortho"])
    def test_gradcheck(self, dct_type, norm):
        """IDCT gradient should pass numerical gradient check."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            lambda inp: inverse_cosine_transform(
                inp, type=dct_type, norm=norm
            ),
            (x,),
            raise_exception=True,
        )


class TestCosineTransformMeta:
    """Test DCT with meta tensors."""

    def test_meta_tensor_shape(self):
        """Meta tensor should produce correct output shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = cosine_transform(x, type=2)
        assert y.shape == torch.Size([16])
        assert y.device.type == "meta"

    def test_meta_tensor_with_n(self):
        """Meta tensor with n parameter should produce correct shape."""
        x = torch.empty(16, device="meta", dtype=torch.float64)
        y = cosine_transform(x, n=32, type=2)
        assert y.shape == torch.Size([32])

    def test_meta_tensor_multi_dim(self):
        """Meta tensor with multi-dim DCT."""
        x = torch.empty(16, 32, device="meta", dtype=torch.float64)
        y = cosine_transform(x, dim=(-2, -1))
        assert y.shape == torch.Size([16, 32])


class TestCosineTransformDevice:
    """Test DCT on different devices."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """DCT should work on CUDA tensors."""
        x = torch.randn(16, dtype=torch.float64, device="cuda")
        y = cosine_transform(x, type=2)
        assert y.device.type == "cuda"

        # Round-trip should work
        x_rec = inverse_cosine_transform(y, type=2)
        assert torch.allclose(x_rec.cpu(), x.cpu(), atol=1e-10)


class TestCosineTransformNormalization:
    """Tests for normalization modes."""

    def test_backward_norm(self):
        """Test backward normalization."""
        x = torch.randn(32, dtype=torch.float64)
        X = cosine_transform(x, norm="backward")
        # Compare with scipy
        X_scipy = scipy_fft.dct(x.numpy(), type=2, norm=None)
        assert np.allclose(X.numpy(), X_scipy, atol=1e-10)

    def test_ortho_norm(self):
        """Test ortho normalization."""
        x = torch.randn(32, dtype=torch.float64)
        X = cosine_transform(x, norm="ortho")
        # Compare with scipy
        X_scipy = scipy_fft.dct(x.numpy(), type=2, norm="ortho")
        assert np.allclose(X.numpy(), X_scipy, atol=1e-10)

    def test_forward_norm(self):
        """Test forward normalization."""
        x = torch.randn(32, dtype=torch.float64)
        X = cosine_transform(x, norm="forward")
        # Compare with scipy
        X_scipy = scipy_fft.dct(x.numpy(), type=2, norm="forward")
        assert np.allclose(X.numpy(), X_scipy, atol=1e-10)


class TestCosineTransformBackwardCompatibility:
    """Tests for backward compatibility with fourier_cosine_transform."""

    def test_alias_exists(self):
        """fourier_cosine_transform should be an alias."""
        assert fourier_cosine_transform is cosine_transform

    def test_inverse_alias_exists(self):
        """inverse_fourier_cosine_transform should be an alias."""
        assert inverse_fourier_cosine_transform is inverse_cosine_transform

    def test_old_api_works(self):
        """Old API should still work."""
        x = torch.randn(32)
        X = fourier_cosine_transform(x, type=2, norm="ortho")
        x_rec = inverse_fourier_cosine_transform(X, type=2, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)


class TestCosineTransformParameterOrder:
    """Tests for new parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        x = torch.randn(32)
        # This should work
        X = cosine_transform(x, type=2, dim=-1, n=64, norm="ortho")
        assert X.shape == torch.Size([64])

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        x = torch.randn(32)
        # This should fail - type should be keyword only
        with pytest.raises(TypeError):
            cosine_transform(x, 2)  # type: ignore


class TestInverseCosineTransformBasic:
    """Basic functionality tests for inverse_cosine_transform."""

    def test_basic_idct2(self):
        """Test basic IDCT-II (default)."""
        X = torch.randn(32)
        x = inverse_cosine_transform(X)
        assert x.shape == torch.Size([32])

    def test_complex_input_raises(self):
        """IDCT should raise error for complex input."""
        X = torch.randn(32, dtype=torch.complex64)
        with pytest.raises(ValueError, match="real"):
            inverse_cosine_transform(X)

    def test_output_is_real(self):
        """IDCT output should be real-valued."""
        X = torch.randn(16, dtype=torch.float64)
        for t in [1, 2, 3, 4]:
            x = inverse_cosine_transform(X, type=t)
            assert x.dtype == torch.float64
            assert not x.is_complex()


class TestInverseCosineTransformMultiDim:
    """Tests for multi-dimensional IDCT."""

    def test_2d_idct(self):
        """Test 2D IDCT with dim tuple."""
        X = torch.randn(16, 32)
        x = inverse_cosine_transform(X, dim=(-2, -1))
        assert x.shape == torch.Size([16, 32])

    def test_2d_with_n(self):
        """Test 2D IDCT with n parameter."""
        X = torch.randn(16, 16)
        x = inverse_cosine_transform(X, dim=(-2, -1), n=(32, 32))
        assert x.shape == torch.Size([32, 32])


class TestInverseCosineTransformPadding:
    """Tests for inverse transform padding modes."""

    def test_constant_padding_inverse(self):
        """Test constant padding for inverse."""
        X = torch.randn(32)
        x = inverse_cosine_transform(X, n=64, padding_mode="constant")
        assert x.shape == torch.Size([64])

    def test_linear_padding_inverse(self):
        """Test linear padding for inverse."""
        X = torch.randn(32)
        x = inverse_cosine_transform(X, n=64, padding_mode="linear")
        assert x.shape == torch.Size([64])

    def test_invalid_padding_mode_inverse(self):
        """Test that invalid padding mode raises error for inverse."""
        X = torch.randn(32)
        with pytest.raises(ValueError, match="padding_mode"):
            inverse_cosine_transform(X, n=64, padding_mode="invalid")


class TestInverseCosineTransformWindow:
    """Tests for inverse transform windowing."""

    def test_with_hann_window_inverse(self):
        """Test with Hann window for inverse (applied after IDCT)."""
        X = torch.randn(32)
        window = torch.hann_window(32)
        x = inverse_cosine_transform(X, window=window)

        # Compare to manual windowing (after transform)
        expected = inverse_cosine_transform(X) * window
        assert torch.allclose(x, expected)


class TestInverseCosineTransformExplicitPadding:
    """Tests for inverse transform explicit padding parameter."""

    def test_explicit_padding_1d_inverse(self):
        """Test explicit padding for 1D inverse."""
        X = torch.randn(32)
        x = inverse_cosine_transform(X, padding=(8, 8))
        assert x.shape == torch.Size([48])

    def test_explicit_padding_multi_dim_inverse(self):
        """Test explicit padding for multi-dim inverse."""
        X = torch.randn(16, 16)
        x = inverse_cosine_transform(X, dim=(-2, -1), padding=((4, 4), (4, 4)))
        assert x.shape == torch.Size([24, 24])


class TestCosineTransformVmap:
    """Tests for torch.vmap support."""

    def test_vmap_basic(self):
        """Test vmap batches correctly."""
        x = torch.randn(8, 32, dtype=torch.float64)

        # Batched call
        y_batched = cosine_transform(x, type=2, dim=-1)

        # vmap call
        def dct_single(xi):
            return cosine_transform(xi, type=2)

        y_vmap = torch.vmap(dct_single)(x)

        assert torch.allclose(y_batched, y_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Test nested vmap."""
        x = torch.randn(4, 3, 16, dtype=torch.float64)

        def dct_single(xi):
            return cosine_transform(xi, type=2)

        y_vmap = torch.vmap(torch.vmap(dct_single))(x)

        assert y_vmap.shape == (4, 3, 16)


class TestCosineTransformCompile:
    """Tests for torch.compile support."""

    def test_compile_basic(self):
        """Test torch.compile works."""
        x = torch.randn(32, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_dct(x):
            return cosine_transform(x, type=2, norm="ortho")

        y_compiled = compiled_dct(x)
        y_eager = cosine_transform(x, type=2, norm="ortho")

        assert torch.allclose(y_compiled, y_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """Test torch.compile with gradient computation."""
        x = torch.randn(16, dtype=torch.float64, requires_grad=True)

        @torch.compile(fullgraph=True)
        def compiled_dct(x):
            return cosine_transform(x, type=2, norm="ortho")

        y = compiled_dct(x)
        loss = y.abs().sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestInverseCosineTransformVmap:
    """Tests for torch.vmap support for inverse transform."""

    def test_vmap_basic(self):
        """Test vmap batches correctly."""
        X = torch.randn(8, 32, dtype=torch.float64)

        # Batched call
        x_batched = inverse_cosine_transform(X, type=2, dim=-1)

        # vmap call
        def idct_single(Xi):
            return inverse_cosine_transform(Xi, type=2)

        x_vmap = torch.vmap(idct_single)(X)

        assert torch.allclose(x_batched, x_vmap, atol=1e-10)


class TestInverseCosineTransformCompile:
    """Tests for torch.compile support for inverse transform."""

    def test_compile_basic(self):
        """Test torch.compile works."""
        X = torch.randn(32, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_idct(X):
            return inverse_cosine_transform(X, type=2, norm="ortho")

        x_compiled = compiled_idct(X)
        x_eager = inverse_cosine_transform(X, type=2, norm="ortho")

        assert torch.allclose(x_compiled, x_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """Test torch.compile with gradient computation."""
        X = torch.randn(16, dtype=torch.float64, requires_grad=True)

        @torch.compile(fullgraph=True)
        def compiled_idct(X):
            return inverse_cosine_transform(X, type=2, norm="ortho")

        x = compiled_idct(X)
        loss = x.abs().sum()
        loss.backward()

        assert X.grad is not None
        assert X.grad.shape == X.shape


class TestCosineTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_input(self):
        """Cosine transform of single element should work."""
        x = torch.tensor([3.0], dtype=torch.float64)
        X = cosine_transform(x, type=2)
        assert X.shape == torch.Size([1])
        assert torch.isfinite(X).all()

    def test_zeros_input(self):
        """Cosine transform of zeros should return zeros."""
        x = torch.zeros(16, dtype=torch.float64)
        X = cosine_transform(x, type=2)
        assert torch.allclose(X, torch.zeros_like(X))

    @pytest.mark.parametrize("dct_type", [1, 2, 3, 4])
    def test_all_types_work(self, dct_type):
        """All DCT types should work."""
        x = torch.randn(16, dtype=torch.float64)
        X = cosine_transform(x, type=dct_type)
        assert X.shape == x.shape

    def test_constant_input(self):
        """Cosine transform of constant has energy at DC."""
        x = torch.ones(16, dtype=torch.float64)
        X = cosine_transform(x, type=2, norm="ortho")
        # First coefficient should dominate
        assert X[0].abs() > X[1:].abs().max() * 10
