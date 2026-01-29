"""Tests for inverse_gabor_transform."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.transform import (
    gabor_transform,
    inverse_gabor_transform,
    inverse_short_time_fourier_transform,
)


class TestInverseGaborTransformBasic:
    """Tests for basic inverse Gabor transform functionality."""

    def test_basic_output_shape(self):
        """Test basic inverse Gabor transform output shape."""
        x = torch.randn(256)
        G = gabor_transform(x, sigma=0.1, n_fft=64)
        x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)

        assert x_rec.shape == (256,)

    def test_real_output(self):
        """Inverse Gabor transform output should be real for real input signal."""
        x = torch.randn(128)
        G = gabor_transform(x, sigma=0.1, n_fft=32)
        x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=32, length=128)

        # Output should be real
        assert not x_rec.is_complex()
        assert x_rec.dtype == x.dtype

    def test_batched_input(self):
        """Test inverse Gabor transform with batched input."""
        x = torch.randn(4, 256)
        G = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)
        x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)

        # Should preserve batch dimension
        assert x_rec.ndim == 2
        assert x_rec.shape[0] == 4  # batch size preserved
        assert x_rec.shape[1] == 256  # signal length

    def test_multi_batch_dims(self):
        """Test inverse Gabor transform with multiple batch dimensions."""
        x = torch.randn(2, 3, 256)
        G = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)
        x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)

        # Should preserve all batch dimensions
        assert x_rec.ndim == 3
        assert x_rec.shape[0] == 2
        assert x_rec.shape[1] == 3
        assert x_rec.shape[2] == 256

    def test_custom_hop_length(self):
        """Test inverse Gabor transform with custom hop_length."""
        x = torch.randn(256)
        G = gabor_transform(x, sigma=0.1, n_fft=64, hop_length=32)
        x_rec = inverse_gabor_transform(
            G, sigma=0.1, n_fft=64, hop_length=32, length=256
        )

        assert x_rec.shape == (256,)


class TestInverseGaborTransformRoundtrip:
    """Tests for roundtrip reconstruction (critical tests)."""

    def test_roundtrip_basic(self):
        """Test basic roundtrip: gabor -> inverse_gabor should reconstruct signal."""
        x = torch.randn(1024)
        sigma = 0.1
        G = gabor_transform(x, sigma=sigma, n_fft=256)
        x_rec = inverse_gabor_transform(G, sigma=sigma, n_fft=256, length=1024)

        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_roundtrip_different_sigma(self):
        """Test roundtrip with different sigma values."""
        x = torch.randn(512)

        for sigma in [0.05, 0.1, 0.2, 0.3]:
            G = gabor_transform(x, sigma=sigma, n_fft=128)
            x_rec = inverse_gabor_transform(
                G, sigma=sigma, n_fft=128, length=512
            )
            assert torch.allclose(x, x_rec, atol=1e-4), (
                f"Failed for sigma={sigma}"
            )

    def test_roundtrip_batched(self):
        """Test roundtrip with batched input."""
        x = torch.randn(4, 512)
        sigma = 0.1
        G = gabor_transform(x, sigma=sigma, n_fft=128, dim=-1)
        x_rec = inverse_gabor_transform(G, sigma=sigma, n_fft=128, length=512)

        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_roundtrip_different_n_fft(self):
        """Test roundtrip with different n_fft values."""
        x = torch.randn(512)
        sigma = 0.1

        for n_fft in [64, 128, 256]:
            G = gabor_transform(x, sigma=sigma, n_fft=n_fft)
            x_rec = inverse_gabor_transform(
                G, sigma=sigma, n_fft=n_fft, length=512
            )
            assert torch.allclose(x, x_rec, atol=1e-4), (
                f"Failed for n_fft={n_fft}"
            )

    def test_roundtrip_with_hop_length(self):
        """Test roundtrip with custom hop_length."""
        x = torch.randn(512)
        sigma = 0.1
        n_fft = 128
        hop_length = 32

        G = gabor_transform(x, sigma=sigma, n_fft=n_fft, hop_length=hop_length)
        x_rec = inverse_gabor_transform(
            G, sigma=sigma, n_fft=n_fft, hop_length=hop_length, length=512
        )

        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_roundtrip_center_false(self):
        """Test roundtrip with center=False.

        Note: center=False with Gaussian windows can be tricky because the
        COLA condition may not be satisfied at boundaries. We use center=True
        by default which handles this properly. For center=False, we verify
        that the transform and inverse work without error, but perfect
        reconstruction may not be achievable at boundaries.
        """
        x = torch.randn(512)
        sigma = 0.15  # Wider Gaussian for better COLA behavior
        n_fft = 64
        # Use smaller hop_length for better overlap
        hop_length = n_fft // 8

        G = gabor_transform(
            x, sigma=sigma, n_fft=n_fft, hop_length=hop_length, center=False
        )
        x_rec = inverse_gabor_transform(
            G,
            sigma=sigma,
            n_fft=n_fft,
            hop_length=hop_length,
            center=False,
            length=512,
        )

        # With appropriate parameters, we should get reasonable reconstruction
        assert torch.allclose(x, x_rec, atol=1e-3)

    def test_roundtrip_ortho_norm(self):
        """Test roundtrip with ortho normalization."""
        x = torch.randn(512)
        sigma = 0.1
        n_fft = 128

        G = gabor_transform(x, sigma=sigma, n_fft=n_fft, norm="ortho")
        x_rec = inverse_gabor_transform(
            G, sigma=sigma, n_fft=n_fft, norm="ortho", length=512
        )

        assert torch.allclose(x, x_rec, atol=1e-4)


class TestInverseGaborTransformSigma:
    """Tests for sigma parameter (Gaussian window width)."""

    def test_sigma_required(self):
        """Test that sigma parameter is required."""
        G = torch.randn(33, 10, dtype=torch.cfloat)  # Fake Gabor coefficients
        with pytest.raises(TypeError):
            inverse_gabor_transform(G, n_fft=64)  # type: ignore

    def test_mismatched_sigma_gives_wrong_reconstruction(self):
        """Test that using different sigma in inverse gives wrong result."""
        x = torch.randn(512)
        G = gabor_transform(x, sigma=0.1, n_fft=128)

        # Use different sigma for inverse
        x_rec = inverse_gabor_transform(G, sigma=0.2, n_fft=128, length=512)

        # Should NOT reconstruct correctly
        assert not torch.allclose(x, x_rec, atol=1e-3)


class TestInverseGaborTransformNormalization:
    """Tests for normalization modes."""

    def test_backward_norm(self):
        """Test backward normalization (default)."""
        G = torch.randn(17, 10, dtype=torch.cfloat)
        x = inverse_gabor_transform(G, sigma=0.1, n_fft=32, norm="backward")

        # Should not error and produce valid output
        assert not x.is_complex()
        assert not torch.isnan(x).any()

    def test_ortho_norm(self):
        """Test ortho normalization."""
        G = torch.randn(17, 10, dtype=torch.cfloat)
        x = inverse_gabor_transform(G, sigma=0.1, n_fft=32, norm="ortho")

        assert not x.is_complex()
        assert not torch.isnan(x).any()

    def test_forward_norm(self):
        """Test forward normalization."""
        G = torch.randn(17, 10, dtype=torch.cfloat)
        x = inverse_gabor_transform(G, sigma=0.1, n_fft=32, norm="forward")

        assert not x.is_complex()
        assert not torch.isnan(x).any()


class TestInverseGaborTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_complex_input(self):
        """Test gradient correctness for complex input."""
        x = torch.randn(64, dtype=torch.float64)
        G = gabor_transform(x, sigma=0.1, n_fft=16, center=False)
        G = G.detach().clone().requires_grad_(True)

        def inverse_gabor_wrapper(G):
            return inverse_gabor_transform(
                G, sigma=0.1, n_fft=16, center=False
            )

        assert gradcheck(
            inverse_gabor_wrapper, (G,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        G = torch.randn(17, 10, dtype=torch.cfloat, requires_grad=True)
        x = inverse_gabor_transform(G, sigma=0.1, n_fft=32)

        loss = x.abs().sum()
        loss.backward()

        assert G.grad is not None
        assert G.grad.shape == G.shape


class TestInverseGaborTransformMeta:
    """Tests for meta tensor support (shape inference).

    Note: Meta tensor support for ISTFT is limited by torch.istft which
    does not support meta tensors. These tests verify the expected behavior.
    """

    @pytest.mark.skip(reason="torch.istft does not support meta tensors")
    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        G = torch.randn(33, 20, dtype=torch.cfloat, device="meta")
        x = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)

        assert x.device == torch.device("meta")
        assert x.shape == (256,)

    @pytest.mark.skip(reason="torch.istft does not support meta tensors")
    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        G = torch.randn(4, 33, 20, dtype=torch.cfloat, device="meta")
        x = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)

        assert x.device == torch.device("meta")
        assert x.shape == (4, 256)


class TestInverseGaborTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(256, device="cuda")
        G = gabor_transform(x, sigma=0.1, n_fft=64)
        x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)

        assert x_rec.device.type == "cuda"
        assert not x_rec.is_complex()


class TestInverseGaborTransformParameterOrder:
    """Tests for parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        G = torch.randn(17, 10, dtype=torch.cfloat)
        # This should work
        x = inverse_gabor_transform(
            G, sigma=0.1, hop_length=8, n_fft=32, norm="ortho"
        )
        assert not x.is_complex()

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        G = torch.randn(17, 10, dtype=torch.cfloat)
        # This should fail - sigma should be keyword only
        with pytest.raises(TypeError):
            inverse_gabor_transform(G, 0.1)  # type: ignore


class TestInverseGaborTransformUsesISTFT:
    """Tests verifying inverse Gabor transform uses ISTFT internally."""

    def test_matches_istft_with_gaussian_window(self):
        """Test that inverse Gabor matches ISTFT with Gaussian window."""
        torch.manual_seed(42)
        x = torch.randn(256)
        n_fft = 64
        sigma = 0.1

        # Compute Gabor transform
        G = gabor_transform(x, sigma=sigma, n_fft=n_fft)

        # Inverse Gabor transform
        x_gabor = inverse_gabor_transform(
            G, sigma=sigma, n_fft=n_fft, length=256
        )

        # Manually create Gaussian window (same as gabor_transform)
        sigma_samples = sigma * n_fft
        t = torch.arange(n_fft, dtype=x.dtype, device=x.device)
        t = t - (n_fft - 1) / 2.0
        window = torch.exp(-0.5 * (t / sigma_samples) ** 2)
        window = window / window.sum() * (n_fft**0.5)

        # Compute ISTFT with same window
        x_istft = inverse_short_time_fourier_transform(
            G, window=window, n_fft=n_fft, length=256
        )

        # They should match
        assert torch.allclose(x_gabor, x_istft, atol=1e-5)

    def test_istft_parameters_passed_through(self):
        """Test that ISTFT parameters are correctly passed through."""
        torch.manual_seed(42)
        x = torch.randn(256)
        n_fft = 64
        # Use default hop_length (n_fft // 4) which satisfies COLA
        hop_length = n_fft // 4
        sigma = 0.1

        G = gabor_transform(
            x,
            sigma=sigma,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
            norm="ortho",
        )

        x1 = inverse_gabor_transform(
            G,
            sigma=sigma,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
            norm="ortho",
            length=256,
        )

        # Manually create Gaussian window
        sigma_samples = sigma * n_fft
        t = torch.arange(n_fft, dtype=x.dtype, device=x.device)
        t = t - (n_fft - 1) / 2.0
        window = torch.exp(-0.5 * (t / sigma_samples) ** 2)
        window = window / window.sum() * (n_fft**0.5)

        x2 = inverse_short_time_fourier_transform(
            G,
            window=window,
            n_fft=n_fft,
            hop_length=hop_length,
            center=True,
            norm="ortho",
            length=256,
        )

        assert torch.allclose(x1, x2, atol=1e-5)


class TestInverseGaborTransformLengthParameter:
    """Tests for length and n parameters."""

    def test_length_parameter(self):
        """Test the length parameter for controlling output size."""
        x = torch.randn(256)
        G = gabor_transform(x, sigma=0.1, n_fft=64)

        x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)
        assert x_rec.shape == (256,)

    def test_n_parameter_alias(self):
        """Test that n parameter works as alias for length."""
        x = torch.randn(256)
        G = gabor_transform(x, sigma=0.1, n_fft=64)

        x_rec = inverse_gabor_transform(G, sigma=0.1, n_fft=64, n=256)
        assert x_rec.shape == (256,)

    def test_length_takes_precedence_over_n(self):
        """Test that length takes precedence over n if both provided."""
        G = torch.randn(33, 20, dtype=torch.cfloat)

        x = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=300, n=200)
        assert x.shape == (300,)

    def test_without_length_infers_from_input(self):
        """Test that without length parameter, size is inferred."""
        G = torch.randn(33, 20, dtype=torch.cfloat)
        x = inverse_gabor_transform(G, sigma=0.1, n_fft=64)

        # Should produce some output (inferred from input)
        assert x.ndim == 1
        assert x.shape[0] > 0


class TestInverseGaborTransformGaussianWindow:
    """Tests verifying the Gaussian window is correctly reconstructed."""

    def test_window_matches_forward_transform(self):
        """Test that the window used matches the forward Gabor transform window."""
        torch.manual_seed(42)
        x = torch.randn(256)
        n_fft = 64
        sigma = 0.15

        # Roundtrip should work perfectly if windows match
        G = gabor_transform(x, sigma=sigma, n_fft=n_fft)
        x_rec = inverse_gabor_transform(
            G, sigma=sigma, n_fft=n_fft, length=256
        )

        # Perfect reconstruction requires matching windows
        assert torch.allclose(x, x_rec, atol=1e-4)

    def test_different_window_gives_different_result(self):
        """Test that different sigma produces different reconstruction."""
        x = torch.randn(256)
        n_fft = 64

        G = gabor_transform(x, sigma=0.1, n_fft=n_fft)

        # Same sigma should reconstruct
        x_same = inverse_gabor_transform(G, sigma=0.1, n_fft=n_fft, length=256)
        # Different sigma should not reconstruct perfectly
        x_diff = inverse_gabor_transform(G, sigma=0.2, n_fft=n_fft, length=256)

        assert torch.allclose(x, x_same, atol=1e-4)
        assert not torch.allclose(x, x_diff, atol=1e-3)


class TestInverseGaborTransformVmap:
    """Tests for torch.vmap support."""

    def test_vmap_basic(self):
        """Test vmap batches correctly."""
        x = torch.randn(8, 256, dtype=torch.float64)
        G = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)

        # Batched call
        x_batched = inverse_gabor_transform(G, sigma=0.1, n_fft=64, length=256)

        # vmap call
        def igabor_single(gi):
            return inverse_gabor_transform(gi, sigma=0.1, n_fft=64, length=256)

        x_vmap = torch.vmap(igabor_single)(G)

        assert torch.allclose(x_batched, x_vmap, atol=1e-10)


class TestInverseGaborTransformCompile:
    """Tests for torch.compile support."""

    @pytest.mark.skip(reason="torch.istft has meta kernel stride issues")
    def test_compile_basic(self):
        """Test torch.compile works."""
        G = torch.randn(33, 20, dtype=torch.complex128)

        @torch.compile(fullgraph=True)
        def compiled_igabor(g):
            return inverse_gabor_transform(g, sigma=0.1, n_fft=64)

        x_compiled = compiled_igabor(G)
        x_eager = inverse_gabor_transform(G, sigma=0.1, n_fft=64)

        assert torch.allclose(x_compiled, x_eager, atol=1e-10)

    @pytest.mark.skip(reason="torch.istft has meta kernel stride issues")
    def test_compile_with_grad(self):
        """Test torch.compile with gradient computation."""
        G = torch.randn(33, 20, dtype=torch.complex128, requires_grad=True)

        @torch.compile(fullgraph=True)
        def compiled_igabor(g):
            return inverse_gabor_transform(g, sigma=0.1, n_fft=64)

        x = compiled_igabor(G)
        loss = x.abs().sum()
        loss.backward()

        assert G.grad is not None
        assert G.grad.shape == G.shape
