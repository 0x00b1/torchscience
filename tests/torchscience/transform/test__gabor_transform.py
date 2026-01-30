"""Tests for gabor_transform."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import (
    gabor_transform,
    short_time_fourier_transform,
)


class TestGaborTransformBasic:
    """Tests for basic Gabor transform functionality."""

    def test_basic_output_shape(self):
        """Test basic Gabor transform output shape for real input."""
        x = torch.randn(256)
        X = gabor_transform(x, sigma=0.1, n_fft=64)

        # For real input with n_fft=64, output freq bins = 64//2 + 1 = 33
        # hop_length defaults to 64//4 = 16
        # num_frames = (256 + 64) // 16 = 20 (with center=True padding)
        assert X.shape[0] == 33  # freq bins
        assert X.ndim == 2

    def test_complex_output(self):
        """Gabor transform output should always be complex."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32)

        assert X.is_complex()

    def test_batched_input(self):
        """Test Gabor transform with batched input."""
        x = torch.randn(4, 256)
        X = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)

        # Should preserve batch dimension
        assert X.ndim == 3
        assert X.shape[0] == 4  # batch size preserved
        assert X.shape[1] == 33  # freq bins

    def test_multi_batch_dims(self):
        """Test Gabor transform with multiple batch dimensions."""
        x = torch.randn(2, 3, 256)
        X = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)

        # Should preserve all batch dimensions
        assert X.ndim == 4
        assert X.shape[0] == 2
        assert X.shape[1] == 3
        assert X.shape[2] == 33  # freq bins

    def test_custom_hop_length(self):
        """Test Gabor transform with custom hop_length."""
        x = torch.randn(256)
        X = gabor_transform(x, sigma=0.1, n_fft=64, hop_length=32)

        # num_frames changes with hop_length
        assert X.shape[0] == 33  # freq bins unchanged
        assert X.ndim == 2


class TestGaborTransformSigma:
    """Tests for sigma parameter (Gaussian window width)."""

    def test_sigma_required(self):
        """Test that sigma parameter is required."""
        x = torch.randn(128)
        with pytest.raises(TypeError):
            gabor_transform(x, n_fft=32)  # type: ignore

    def test_different_sigma_produces_different_results(self):
        """Test that different sigma values produce different results."""
        x = torch.randn(256)
        X1 = gabor_transform(x, sigma=0.05, n_fft=64)
        X2 = gabor_transform(x, sigma=0.2, n_fft=64)

        # Results should be different
        assert not torch.allclose(X1, X2)

    def test_small_sigma_more_time_localized(self):
        """Test that smaller sigma gives more time localization.

        With smaller sigma, the window is narrower, so each frame
        captures less of the signal. We verify by checking that
        narrow sigma (more time-localized) has different energy
        distribution than wide sigma.
        """
        torch.manual_seed(42)
        # Create a signal with a sharp transient
        x = torch.zeros(256)
        x[128] = 10.0  # impulse at center

        X_narrow = gabor_transform(x, sigma=0.05, n_fft=64)
        X_wide = gabor_transform(x, sigma=0.3, n_fft=64)

        # Both should have valid output
        assert X_narrow.is_complex()
        assert X_wide.is_complex()
        # Results should differ due to different time-frequency tradeoff
        assert not torch.allclose(X_narrow, X_wide)

    def test_sigma_affects_window_width(self):
        """Test that sigma controls the Gaussian window width."""
        x = torch.randn(128)

        # Very small sigma = very narrow window
        X_small = gabor_transform(x, sigma=0.01, n_fft=32)
        # Large sigma = wide window (approaches rectangular)
        X_large = gabor_transform(x, sigma=1.0, n_fft=32)

        assert X_small.is_complex()
        assert X_large.is_complex()
        assert not torch.allclose(X_small, X_large)


class TestGaborTransformNormalization:
    """Tests for normalization modes."""

    def test_backward_norm(self):
        """Test backward normalization (default)."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, norm="backward")

        # Should not error and produce valid output
        assert X.is_complex()
        assert not torch.isnan(X).any()

    def test_ortho_norm(self):
        """Test ortho normalization."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, norm="ortho")

        assert X.is_complex()
        assert not torch.isnan(X).any()

    def test_forward_norm(self):
        """Test forward normalization."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, norm="forward")

        assert X.is_complex()
        assert not torch.isnan(X).any()

    def test_all_norms_differ(self):
        """Test that different norm modes produce different results."""
        x = torch.randn(128)
        X_backward = gabor_transform(x, sigma=0.1, n_fft=32, norm="backward")
        X_ortho = gabor_transform(x, sigma=0.1, n_fft=32, norm="ortho")
        X_forward = gabor_transform(x, sigma=0.1, n_fft=32, norm="forward")

        assert not torch.allclose(X_backward, X_ortho)
        assert not torch.allclose(X_backward, X_forward)
        assert not torch.allclose(X_ortho, X_forward)


class TestGaborTransformPadding:
    """Tests for padding modes."""

    def test_reflect_padding(self):
        """Test reflect padding mode."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, padding_mode="reflect")

        assert X.is_complex()

    def test_constant_padding(self):
        """Test constant (zero) padding mode."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, padding_mode="constant")

        assert X.is_complex()

    def test_replicate_padding(self):
        """Test replicate padding mode."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, padding_mode="replicate")

        assert X.is_complex()

    def test_circular_padding(self):
        """Test circular padding mode."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, padding_mode="circular")

        assert X.is_complex()

    def test_center_false(self):
        """Test with center=False (no centering padding)."""
        x = torch.randn(128)
        X_centered = gabor_transform(x, sigma=0.1, n_fft=32, center=True)
        X_not_centered = gabor_transform(x, sigma=0.1, n_fft=32, center=False)

        # Not centered should have fewer frames
        assert X_not_centered.shape[-1] < X_centered.shape[-1]

    def test_explicit_padding(self):
        """Test explicit padding parameter."""
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, padding=16, center=False)

        assert X.is_complex()


class TestGaborTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_real_input(self):
        """Test gradient correctness for real input."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)

        def gabor_wrapper(x):
            return gabor_transform(x, sigma=0.1, n_fft=16, center=False)

        assert gradcheck(gabor_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck_real_input(self):
        """Test second-order gradient correctness for real input."""
        x = torch.randn(32, dtype=torch.float64, requires_grad=True)

        def gabor_wrapper(x):
            return gabor_transform(x, sigma=0.1, n_fft=16, center=False)

        assert gradgradcheck(
            gabor_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        x = torch.randn(128, requires_grad=True)
        X = gabor_transform(x, sigma=0.1, n_fft=32)

        loss = X.abs().sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestGaborTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.randn(256, device="meta")
        X = gabor_transform(x, sigma=0.1, n_fft=64)

        assert X.device == torch.device("meta")
        assert X.shape[0] == 33  # n_fft//2 + 1

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        x = torch.randn(4, 256, device="meta")
        X = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)

        assert X.device == torch.device("meta")
        assert X.shape[0] == 4  # batch preserved
        assert X.shape[1] == 33  # freq bins


class TestGaborTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(256, device="cuda")
        X = gabor_transform(x, sigma=0.1, n_fft=64)

        assert X.device.type == "cuda"
        assert X.is_complex()


class TestGaborTransformParameterOrder:
    """Tests for parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        x = torch.randn(128)
        # This should work
        X = gabor_transform(
            x, sigma=0.1, hop_length=16, n_fft=32, norm="ortho"
        )
        assert X.is_complex()

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        x = torch.randn(128)
        # This should fail - sigma should be keyword only
        with pytest.raises(TypeError):
            gabor_transform(x, 0.1)  # type: ignore


class TestGaborTransformUsesSTFT:
    """Tests verifying Gabor transform uses STFT internally."""

    def test_matches_stft_with_gaussian_window(self):
        """Test that Gabor transform matches STFT with Gaussian window."""
        torch.manual_seed(42)
        x = torch.randn(256)
        n_fft = 64
        sigma = 0.1

        # Compute Gabor transform
        X_gabor = gabor_transform(x, sigma=sigma, n_fft=n_fft)

        # Manually create Gaussian window
        sigma_samples = sigma * n_fft
        t = torch.arange(n_fft, dtype=x.dtype, device=x.device)
        t = t - (n_fft - 1) / 2.0
        window = torch.exp(-0.5 * (t / sigma_samples) ** 2)
        window = window / window.sum() * (n_fft**0.5)

        # Compute STFT with same window
        X_stft = short_time_fourier_transform(x, window=window, n_fft=n_fft)

        # They should match
        assert torch.allclose(X_gabor, X_stft, atol=1e-5)

    def test_stft_parameters_passed_through(self):
        """Test that STFT parameters are correctly passed through."""
        torch.manual_seed(42)
        x = torch.randn(256)
        n_fft = 64
        hop_length = 32
        sigma = 0.1

        X1 = gabor_transform(
            x,
            sigma=sigma,
            n_fft=n_fft,
            hop_length=hop_length,
            center=False,
            norm="ortho",
        )

        # Manually create Gaussian window
        sigma_samples = sigma * n_fft
        t = torch.arange(n_fft, dtype=x.dtype, device=x.device)
        t = t - (n_fft - 1) / 2.0
        window = torch.exp(-0.5 * (t / sigma_samples) ** 2)
        window = window / window.sum() * (n_fft**0.5)

        X2 = short_time_fourier_transform(
            x,
            window=window,
            n_fft=n_fft,
            hop_length=hop_length,
            center=False,
            norm="ortho",
        )

        assert torch.allclose(X1, X2, atol=1e-5)


class TestGaborTransformDim:
    """Tests for dim parameter handling."""

    def test_dim_middle(self):
        """Test Gabor transform on middle dimension."""
        x = torch.randn(4, 256, 3)
        X = gabor_transform(x, sigma=0.1, n_fft=64, dim=1)

        # Signal dim (1) is transformed to (freq, frames) at the end
        # Remaining dims: (4, 3) -> output: (4, 3, freq, frames)
        assert X.shape[0] == 4  # first batch dim preserved
        assert X.shape[1] == 3  # second batch dim (was after signal dim)
        assert X.shape[2] == 33  # freq bins = n_fft//2 + 1

    def test_dim_first(self):
        """Test Gabor transform on first dimension."""
        x = torch.randn(256, 4, 3)
        X = gabor_transform(x, sigma=0.1, n_fft=64, dim=0)

        # Signal dim (0) is transformed to (freq, frames) at the end
        # Remaining dims: (4, 3) -> output: (4, 3, freq, frames)
        assert X.shape[0] == 4
        assert X.shape[1] == 3
        assert X.shape[2] == 33  # freq bins

    def test_negative_dim(self):
        """Test Gabor transform with negative dimension."""
        x = torch.randn(4, 256)
        X = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)

        # Output: (4, freq, frames)
        assert X.shape[0] == 4  # batch preserved
        assert X.shape[1] == 33  # freq bins


class TestGaborTransformNParameter:
    """Tests for n parameter (signal length control)."""

    def test_n_larger_than_input(self):
        """Test with n larger than input size."""
        x = torch.randn(64)
        X = gabor_transform(x, sigma=0.1, n_fft=32, n=128)

        # Should have more frames due to longer signal
        assert X.is_complex()

    def test_n_smaller_than_input(self):
        """Test with n smaller than input size."""
        x = torch.randn(256)
        X = gabor_transform(x, sigma=0.1, n_fft=32, n=64)

        # Should have fewer frames due to truncation
        assert X.is_complex()


class TestGaborTransformGaussianWindow:
    """Tests verifying the Gaussian window is correctly generated."""

    def test_window_is_gaussian_shaped(self):
        """Test that the internal window has Gaussian shape.

        We can verify this by comparing the Gabor output with STFT
        using a manually constructed Gaussian window.
        """
        x = torch.randn(128)
        n_fft = 32
        sigma = 0.15

        # Get Gabor result
        X_gabor = gabor_transform(x, sigma=sigma, n_fft=n_fft)

        # Construct expected Gaussian window
        sigma_samples = sigma * n_fft
        t = torch.arange(n_fft, dtype=x.dtype, device=x.device)
        t = t - (n_fft - 1) / 2.0
        window = torch.exp(-0.5 * (t / sigma_samples) ** 2)
        window = window / window.sum() * (n_fft**0.5)

        # Verify window is centered and bell-shaped
        center = n_fft // 2
        assert window[center] >= window[0]  # peak at center
        assert window[center] >= window[-1]  # peak at center

        # Verify Gabor matches STFT with this window
        X_stft = short_time_fourier_transform(x, window=window, n_fft=n_fft)
        assert torch.allclose(X_gabor, X_stft, atol=1e-5)

    def test_window_normalized(self):
        """Test that the window is properly normalized."""
        # We can verify normalization indirectly by checking energy preservation
        torch.manual_seed(42)
        x = torch.randn(128)
        X = gabor_transform(x, sigma=0.1, n_fft=32, norm="ortho")

        # With ortho normalization, should have reasonable energy
        assert X.abs().mean() > 0
        assert not torch.isinf(X).any()
        assert not torch.isnan(X).any()


class TestGaborTransformVmap:
    """Tests for Gabor transform with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        x = torch.randn(8, 256, dtype=torch.float64)

        # Manual batching
        X_batched = gabor_transform(x, sigma=0.1, n_fft=64, dim=-1)

        # vmap
        def gabor_single(xi):
            return gabor_transform(xi, sigma=0.1, n_fft=64)

        X_vmap = torch.vmap(gabor_single)(x)

        assert torch.allclose(X_batched, X_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        x = torch.randn(4, 4, 128, dtype=torch.float64)

        def gabor_single(xi):
            return gabor_transform(xi, sigma=0.1, n_fft=32)

        X_vmap = torch.vmap(torch.vmap(gabor_single))(x)

        assert X_vmap.ndim == 4  # batch, batch, freq, frames


class TestGaborTransformCompile:
    """Tests for Gabor transform with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        x = torch.randn(256, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_gabor(xi):
            return gabor_transform(xi, sigma=0.1, n_fft=64)

        X_compiled = compiled_gabor(x)
        X_eager = gabor_transform(x, sigma=0.1, n_fft=64)

        assert torch.allclose(X_compiled, X_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        x = torch.randn(128, dtype=torch.float64, requires_grad=True)

        @torch.compile(fullgraph=True)
        def compiled_gabor(xi):
            return gabor_transform(xi, sigma=0.1, n_fft=32, center=False)

        X = compiled_gabor(x)
        X.abs().sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestGaborTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_zeros_input(self):
        """Gabor transform of zeros should return zeros."""
        x = torch.zeros(256, dtype=torch.float64)
        X = gabor_transform(x, sigma=0.1, n_fft=64, center=False)
        assert torch.allclose(X, torch.zeros_like(X))

    def test_constant_input(self):
        """Gabor transform of constant should have energy at DC."""
        x = torch.ones(256, dtype=torch.float64)
        X = gabor_transform(x, sigma=0.1, n_fft=64, center=False)
        # DC component should dominate in each frame
        assert X[0, :].abs().mean() > X[1:, :].abs().mean() * 10

    def test_sigma_parameter(self):
        """Different sigma values should produce different results."""
        x = torch.randn(256, dtype=torch.float64)
        X1 = gabor_transform(x, sigma=0.05, n_fft=64, center=False)
        X2 = gabor_transform(x, sigma=0.2, n_fft=64, center=False)
        # Results should differ due to different window widths
        assert not torch.allclose(X1, X2)

    def test_minimum_length_input(self):
        """Gabor transform should work with minimum valid input length."""
        x = torch.randn(32, dtype=torch.float64)
        X = gabor_transform(x, sigma=0.1, n_fft=32, center=False)
        assert X.ndim == 2
