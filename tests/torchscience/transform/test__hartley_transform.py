"""Tests for hartley_transform."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import hartley_transform


class TestHartleyTransformBasic:
    """Tests for basic hartley_transform functionality."""

    def test_basic(self):
        """Test basic Hartley transform."""
        x = torch.randn(32)
        H = hartley_transform(x)
        assert H.shape == torch.Size([32])
        assert H.dtype == x.dtype  # Real output

    def test_output_is_real(self):
        """Hartley transform output should always be real."""
        x = torch.randn(32)
        H = hartley_transform(x)
        assert not H.is_complex()
        assert H.dtype == x.dtype

    def test_complex_input_raises(self):
        """Test that complex input raises ValueError."""
        x = torch.randn(32, dtype=torch.complex64)
        with pytest.raises(ValueError, match="real"):
            hartley_transform(x)

    def test_complex128_input_raises(self):
        """Test that complex128 input also raises ValueError."""
        x = torch.randn(32, dtype=torch.complex128)
        with pytest.raises(ValueError, match="real"):
            hartley_transform(x)

    def test_shape_preserved(self):
        """Output shape matches input shape along transform dim."""
        x = torch.randn(10, 20, 30)
        H = hartley_transform(x, dim=1)
        assert H.shape == x.shape

    def test_with_n_larger(self):
        """Test padding when n > input size."""
        x = torch.randn(64)
        H = hartley_transform(x, n=128)
        assert H.shape == torch.Size([128])

    def test_with_n_smaller(self):
        """Test truncation when n < input size."""
        x = torch.randn(128)
        H = hartley_transform(x, n=64)
        assert H.shape == torch.Size([64])


class TestHartleyTransformSelfInverse:
    """Tests for Hartley transform self-inverse property."""

    def test_self_inverse_ortho(self):
        """Hartley is self-inverse with ortho normalization."""
        x = torch.randn(32)
        H = hartley_transform(x, norm="ortho")
        x_rec = hartley_transform(H, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_self_inverse_ortho_double(self):
        """Hartley self-inverse with double precision."""
        x = torch.randn(32, dtype=torch.float64)
        H = hartley_transform(x, norm="ortho")
        x_rec = hartley_transform(H, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-10)

    def test_self_inverse_ortho_batched(self):
        """Hartley self-inverse with batched input."""
        x = torch.randn(4, 32)
        H = hartley_transform(x, dim=-1, norm="ortho")
        x_rec = hartley_transform(H, dim=-1, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)


class TestHartleyTransformRelationToFFT:
    """Tests for Hartley-FFT relationship."""

    def test_hartley_fft_relationship(self):
        """Verify H = Re(F) - Im(F)."""
        x = torch.randn(32)
        H = hartley_transform(x)
        F = torch.fft.fft(x)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_hartley_fft_relationship_double(self):
        """Verify H = Re(F) - Im(F) with double precision."""
        x = torch.randn(32, dtype=torch.float64)
        H = hartley_transform(x)
        F = torch.fft.fft(x)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_hartley_fft_relationship_batched(self):
        """Verify H = Re(F) - Im(F) with batched input."""
        x = torch.randn(4, 8, 32)
        H = hartley_transform(x, dim=-1)
        F = torch.fft.fft(x, dim=-1)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_hartley_fft_relationship_all_norms(self):
        """Verify relationship holds for all normalization modes."""
        x = torch.randn(32)
        for norm in ["forward", "backward", "ortho"]:
            H = hartley_transform(x, norm=norm)
            F = torch.fft.fft(x, norm=norm)
            expected = F.real - F.imag
            assert torch.allclose(H, expected), f"Mismatch for norm={norm}"


class TestHartleyTransformMultiDim:
    """Tests for multi-dimensional Hartley transform."""

    def test_2d(self):
        """Test 2D Hartley transform."""
        x = torch.randn(16, 32)
        H = hartley_transform(x, dim=(-2, -1))
        assert H.shape == torch.Size([16, 32])
        assert not H.is_complex()

    def test_2d_self_inverse(self):
        """Test 2D Hartley self-inverse."""
        x = torch.randn(16, 32)
        H = hartley_transform(x, dim=(-2, -1), norm="ortho")
        x_rec = hartley_transform(H, dim=(-2, -1), norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_2d_fft_relationship(self):
        """Verify 2D H = Re(F) - Im(F)."""
        x = torch.randn(16, 32)
        H = hartley_transform(x, dim=(-2, -1))
        F = torch.fft.fft2(x)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_nd_transform(self):
        """Test nD Hartley transform."""
        x = torch.randn(4, 8, 16)
        H = hartley_transform(x, dim=(0, 1, 2))
        F = torch.fft.fftn(x)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_n_with_multi_dim(self):
        """Test n parameter with multi-dim."""
        x = torch.randn(16, 16)
        H = hartley_transform(x, dim=(-2, -1), n=(32, 32))
        assert H.shape == torch.Size([32, 32])

    def test_n_tuple_truncation(self):
        """Test n parameter for truncation with multi-dim."""
        x = torch.randn(32, 32)
        H = hartley_transform(x, dim=(-2, -1), n=(16, 16))
        assert H.shape == torch.Size([16, 16])

    def test_single_dim_as_tuple(self):
        """Test single dim provided as tuple."""
        x = torch.randn(32, 64)
        H = hartley_transform(x, dim=(-1,))
        F = torch.fft.fft(x, dim=-1)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_n_tuple_length_mismatch_raises(self):
        """Test that mismatched n and dim tuple lengths raise error."""
        x = torch.randn(16, 16)
        with pytest.raises(ValueError, match="length"):
            hartley_transform(x, dim=(-2, -1), n=(32,))

    def test_2d_with_batch_dim(self):
        """Test 2D Hartley preserves batch dimensions."""
        x = torch.randn(4, 8, 16, 32)
        H = hartley_transform(x, dim=(-2, -1))
        F = torch.fft.fft2(x, dim=(-2, -1))
        expected = F.real - F.imag
        assert torch.allclose(H, expected)
        assert H.shape == x.shape


class TestHartleyTransformNormalization:
    """Tests for normalization modes."""

    def test_backward_norm(self):
        """Test backward normalization."""
        x = torch.randn(32)
        H = hartley_transform(x, norm="backward")
        F = torch.fft.fft(x, norm="backward")
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_ortho_norm(self):
        """Test ortho normalization."""
        x = torch.randn(32)
        H = hartley_transform(x, norm="ortho")
        F = torch.fft.fft(x, norm="ortho")
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_forward_norm(self):
        """Test forward normalization."""
        x = torch.randn(32)
        H = hartley_transform(x, norm="forward")
        F = torch.fft.fft(x, norm="forward")
        expected = F.real - F.imag
        assert torch.allclose(H, expected)


class TestHartleyTransformPadding:
    """Tests for padding modes."""

    def test_constant_padding(self):
        """Test constant (zero) padding."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="constant")
        assert H.shape == torch.Size([64])

    def test_reflect_padding(self):
        """Test reflect padding."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="reflect")
        assert H.shape == torch.Size([64])

    def test_replicate_padding(self):
        """Test replicate padding."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="replicate")
        assert H.shape == torch.Size([64])

    def test_circular_padding(self):
        """Test circular padding."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="circular")
        assert H.shape == torch.Size([64])

    def test_invalid_padding_mode(self):
        """Test that invalid padding mode raises error."""
        x = torch.randn(32)
        with pytest.raises(ValueError, match="padding_mode"):
            hartley_transform(x, n=64, padding_mode="invalid")

    def test_linear_padding(self):
        """Test linear extrapolation padding."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="linear")
        assert H.shape == torch.Size([64])

    def test_smooth_padding(self):
        """Test smooth padding mode."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="smooth")
        assert H.shape == torch.Size([64])

    def test_polynomial_padding(self):
        """Test polynomial padding mode."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="polynomial")
        assert H.shape == torch.Size([64])

    def test_spline_padding(self):
        """Test spline padding mode."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="spline")
        assert H.shape == torch.Size([64])

    def test_padding_order_for_polynomial(self):
        """Test padding_order parameter for polynomial mode."""
        x = torch.randn(32)
        H = hartley_transform(
            x, n=64, padding_mode="polynomial", padding_order=2
        )
        assert H.shape == torch.Size([64])

    def test_reflect_odd_padding(self):
        """Test reflect_odd (antisymmetric) padding mode."""
        x = torch.randn(32)
        H = hartley_transform(x, n=64, padding_mode="reflect_odd")
        assert H.shape == torch.Size([64])


class TestHartleyTransformExplicitPadding:
    """Tests for explicit padding parameter."""

    def test_explicit_padding_1d(self):
        """Test explicit padding for 1D."""
        x = torch.randn(32)
        # Pad 8 on left, 8 on right
        H = hartley_transform(x, padding=(8, 8))
        assert H.shape == torch.Size([48])

    def test_explicit_padding_asymmetric(self):
        """Test asymmetric explicit padding."""
        x = torch.randn(32)
        H = hartley_transform(x, padding=(4, 12))
        assert H.shape == torch.Size([48])

    def test_explicit_padding_multi_dim(self):
        """Test explicit padding for multi-dim transform."""
        x = torch.randn(16, 16)
        # Pad each dim by (4, 4)
        H = hartley_transform(x, dim=(-2, -1), padding=((4, 4), (4, 4)))
        assert H.shape == torch.Size([24, 24])

    def test_n_overrides_padding(self):
        """Test that n parameter works with explicit padding."""
        x = torch.randn(32)
        # Explicit padding of 8 each side would give 48, but n=64 should override
        H = hartley_transform(x, padding=(8, 8), n=64)
        assert H.shape == torch.Size([64])


class TestHartleyTransformWindow:
    """Tests for windowing."""

    def test_with_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(32)
        window = torch.hann_window(32)
        H = hartley_transform(x, window=window)

        F = torch.fft.fft(x * window)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_with_hamming_window(self):
        """Test with Hamming window."""
        x = torch.randn(32)
        window = torch.hamming_window(32)
        H = hartley_transform(x, window=window)

        F = torch.fft.fft(x * window)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)

    def test_window_size_mismatch_raises(self):
        """Test that window size mismatch raises error."""
        x = torch.randn(32)
        window = torch.hann_window(64)
        with pytest.raises(ValueError, match="window size"):
            hartley_transform(x, window=window)

    def test_window_not_1d_raises(self):
        """Test that non-1D window raises error."""
        x = torch.randn(32)
        window = torch.randn(8, 4)
        with pytest.raises(ValueError, match="1-D"):
            hartley_transform(x, window=window)

    def test_window_multi_dim_raises(self):
        """Test that windowing with multi-dim raises error."""
        x = torch.randn(16, 32)
        window = torch.hann_window(16)
        with pytest.raises(ValueError, match="single-dimension"):
            hartley_transform(x, dim=(-2, -1), window=window)


class TestHartleyTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_real_input(self):
        """Test gradient correctness for real input."""
        x = torch.randn(16, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            hartley_transform, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradgradcheck(self):
        """Test second-order gradient correctness."""
        x = torch.randn(8, dtype=torch.float64, requires_grad=True)
        assert gradgradcheck(
            hartley_transform, (x,), eps=1e-6, atol=1e-3, rtol=1e-2
        )

    def test_gradcheck_batched(self):
        """Test gradient correctness for batched input."""
        x = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
        assert gradcheck(
            hartley_transform, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_gradcheck_multi_dim(self):
        """Test gradient correctness for multi-dim transform."""
        x = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

        def func(t):
            return hartley_transform(t, dim=(-2, -1))

        assert gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)


class TestHartleyTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(32, device="cuda")
        H = hartley_transform(x)

        F = torch.fft.fft(x)
        expected = F.real - F.imag
        assert torch.allclose(H, expected)
        assert H.device == x.device

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_self_inverse(self):
        """Test self-inverse on CUDA."""
        x = torch.randn(32, device="cuda")
        H = hartley_transform(x, norm="ortho")
        x_rec = hartley_transform(H, norm="ortho")
        assert torch.allclose(x, x_rec, atol=1e-5)


class TestHartleyTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.randn(32, device="meta")
        H = hartley_transform(x)

        assert H.shape == torch.Size([32])
        assert H.device == torch.device("meta")

    def test_meta_tensor_with_n(self):
        """Test shape inference with n parameter."""
        x = torch.randn(32, device="meta")
        H = hartley_transform(x, n=64)

        assert H.shape == torch.Size([64])

    def test_meta_tensor_multi_dim(self):
        """Test shape inference for multi-dim."""
        x = torch.randn(16, 32, device="meta")
        H = hartley_transform(x, dim=(-2, -1))

        assert H.shape == torch.Size([16, 32])


class TestHartleyTransformDtype:
    """Tests for dtype handling."""

    def test_float32(self):
        """Test float32 input."""
        x = torch.randn(32, dtype=torch.float32)
        H = hartley_transform(x)
        assert H.dtype == torch.float32

    def test_float64(self):
        """Test float64 input."""
        x = torch.randn(32, dtype=torch.float64)
        H = hartley_transform(x)
        assert H.dtype == torch.float64

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="float16 FFT requires CUDA",
    )
    def test_float16_cuda(self):
        """Test float16 input on CUDA (float16 FFT not supported on CPU)."""
        x = torch.randn(32, dtype=torch.float16, device="cuda")
        H = hartley_transform(x)
        # FFT promotes float16 to float32 internally on CUDA
        assert H.dtype in (torch.float16, torch.float32)


class TestHartleyTransformParameterOrder:
    """Tests for new parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        x = torch.randn(32)
        # This should work
        H = hartley_transform(x, dim=-1, n=64, norm="ortho")
        assert H.shape == torch.Size([64])

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        x = torch.randn(32)
        # This should fail - dim should be keyword only
        with pytest.raises(TypeError):
            hartley_transform(x, -1)  # type: ignore


class TestHartleyTransformOut:
    """Tests for out parameter."""

    def test_out_parameter(self):
        """Test that out parameter works."""
        x = torch.randn(32)
        out = torch.empty(32)
        result = hartley_transform(x, out=out)

        F = torch.fft.fft(x)
        expected = F.real - F.imag

        assert result is out
        assert torch.allclose(out, expected)

    def test_out_parameter_with_n(self):
        """Test out parameter with n."""
        x = torch.randn(32)
        out = torch.empty(64)
        result = hartley_transform(x, n=64, out=out)

        assert result is out
        assert out.shape == torch.Size([64])


class TestHartleyTransformVmap:
    """Tests for Hartley transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        x = torch.randn(8, 64, dtype=torch.float64)

        # Manual batching via dim parameter
        H_batched = hartley_transform(x, dim=-1)

        # vmap
        def hartley_single(xi):
            return hartley_transform(xi)

        H_vmap = torch.vmap(hartley_single)(x)

        assert torch.allclose(H_batched, H_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        x = torch.randn(4, 4, 32, dtype=torch.float64)

        def hartley_single(xi):
            return hartley_transform(xi)

        H_vmap = torch.vmap(torch.vmap(hartley_single))(x)

        assert H_vmap.shape == torch.Size([4, 4, 32])


class TestHartleyTransformCompile:
    """Tests for Hartley transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        x = torch.randn(64, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_hartley(xi):
            return hartley_transform(xi)

        H_compiled = compiled_hartley(x)
        H_eager = hartley_transform(x)

        assert torch.allclose(H_compiled, H_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        x = torch.randn(32, dtype=torch.float64, requires_grad=True)

        @torch.compile(fullgraph=True)
        def compiled_hartley(xi):
            return hartley_transform(xi)

        H = compiled_hartley(x)
        H.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestHartleyTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_input(self):
        """Hartley transform of single element should work."""
        x = torch.tensor([3.0], dtype=torch.float64)
        H = hartley_transform(x)
        assert H.shape == torch.Size([1])
        # Hartley of single element is itself
        assert torch.allclose(H, x)

    def test_zeros_input(self):
        """Hartley transform of zeros should return zeros."""
        x = torch.zeros(32, dtype=torch.float64)
        H = hartley_transform(x)
        assert torch.allclose(H, torch.zeros_like(H))

    def test_constant_input(self):
        """Hartley transform of constant should have energy at DC."""
        x = torch.ones(32, dtype=torch.float64)
        H = hartley_transform(x)
        # DC component should dominate
        assert H[0].abs() > H[1:].abs().max() * 100

    def test_impulse_input(self):
        """Hartley transform of impulse should be constant."""
        x = torch.zeros(32, dtype=torch.float64)
        x[0] = 1.0
        H = hartley_transform(x)
        # All Hartley components should be equal (to 1)
        assert torch.allclose(H, torch.ones_like(H), atol=1e-10)

    def test_power_of_two_size(self):
        """Hartley should work with power-of-two sizes."""
        for n in [2, 4, 8, 16, 32, 64, 128]:
            x = torch.randn(n, dtype=torch.float64)
            H = hartley_transform(x)
            assert H.shape == torch.Size([n])

    def test_non_power_of_two_size(self):
        """Hartley should work with non-power-of-two sizes."""
        for n in [3, 5, 7, 11, 13, 17, 19, 23]:
            x = torch.randn(n, dtype=torch.float64)
            H = hartley_transform(x)
            assert H.shape == torch.Size([n])

    def test_output_is_real(self):
        """Hartley transform output should always be real."""
        x = torch.randn(32, dtype=torch.float64)
        H = hartley_transform(x)
        assert not H.is_complex()
        assert H.dtype == torch.float64
