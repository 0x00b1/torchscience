"""Tests for short_time_fourier_transform."""

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from torchscience.transform import short_time_fourier_transform


class TestShortTimeFourierTransformBasic:
    """Tests for basic STFT functionality."""

    def test_basic_output_shape(self):
        """Test basic STFT output shape for real input."""
        x = torch.randn(256)
        window = torch.hann_window(64)
        X = short_time_fourier_transform(x, window=window)

        # For real input with n_fft=64, output freq bins = 64//2 + 1 = 33
        # hop_length defaults to 64//4 = 16
        # num_frames = (256 + 64) // 16 = 20 (with center=True padding)
        assert X.shape[0] == 33  # freq bins
        assert X.ndim == 2

    def test_complex_output(self):
        """STFT output should always be complex."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window)

        assert X.is_complex()

    def test_batched_input(self):
        """Test STFT with batched input."""
        x = torch.randn(4, 256)
        window = torch.hann_window(64)
        X = short_time_fourier_transform(x, window=window, dim=-1)

        # Should preserve batch dimension
        assert X.ndim == 3
        assert X.shape[0] == 4  # batch size preserved
        assert X.shape[1] == 33  # freq bins

    def test_multi_batch_dims(self):
        """Test STFT with multiple batch dimensions."""
        x = torch.randn(2, 3, 256)
        window = torch.hann_window(64)
        X = short_time_fourier_transform(x, window=window, dim=-1)

        # Should preserve all batch dimensions
        assert X.ndim == 4
        assert X.shape[0] == 2
        assert X.shape[1] == 3
        assert X.shape[2] == 33  # freq bins

    def test_custom_n_fft(self):
        """Test STFT with custom n_fft."""
        x = torch.randn(256)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window, n_fft=64)

        # With n_fft=64, freq bins = 64//2 + 1 = 33
        assert X.shape[0] == 33

    def test_custom_hop_length(self):
        """Test STFT with custom hop_length."""
        x = torch.randn(256)
        window = torch.hann_window(64)
        X = short_time_fourier_transform(x, window=window, hop_length=32)

        # num_frames changes with hop_length
        assert X.shape[0] == 33  # freq bins unchanged
        assert X.ndim == 2


class TestShortTimeFourierTransformNormalization:
    """Tests for normalization modes."""

    def test_backward_norm(self):
        """Test backward normalization (default)."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window, norm="backward")

        # Should not error and produce valid output
        assert X.is_complex()
        assert not torch.isnan(X).any()

    def test_ortho_norm(self):
        """Test ortho normalization."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window, norm="ortho")

        assert X.is_complex()
        assert not torch.isnan(X).any()

    def test_forward_norm(self):
        """Test forward normalization."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window, norm="forward")

        assert X.is_complex()
        assert not torch.isnan(X).any()


class TestShortTimeFourierTransformWindow:
    """Tests for window handling."""

    def test_window_required(self):
        """Test that window parameter is required."""
        x = torch.randn(128)
        with pytest.raises(ValueError, match="window"):
            short_time_fourier_transform(x, window=None)

    def test_window_required_explicit_none(self):
        """Test that explicitly passing window=None raises error."""
        x = torch.randn(128)
        with pytest.raises(ValueError, match="window"):
            short_time_fourier_transform(x, window=None)

    def test_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window)

        assert X.is_complex()

    def test_hamming_window(self):
        """Test with Hamming window."""
        x = torch.randn(128)
        window = torch.hamming_window(32)
        X = short_time_fourier_transform(x, window=window)

        assert X.is_complex()

    def test_blackman_window(self):
        """Test with Blackman window."""
        x = torch.randn(128)
        window = torch.blackman_window(32)
        X = short_time_fourier_transform(x, window=window)

        assert X.is_complex()

    def test_rectangular_window(self):
        """Test with rectangular (ones) window."""
        x = torch.randn(128)
        window = torch.ones(32)
        X = short_time_fourier_transform(x, window=window)

        assert X.is_complex()


class TestShortTimeFourierTransformPadding:
    """Tests for padding modes."""

    def test_reflect_padding(self):
        """Test reflect padding mode."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(
            x, window=window, padding_mode="reflect"
        )

        assert X.is_complex()

    def test_constant_padding(self):
        """Test constant (zero) padding mode."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(
            x, window=window, padding_mode="constant"
        )

        assert X.is_complex()

    def test_replicate_padding(self):
        """Test replicate padding mode."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(
            x, window=window, padding_mode="replicate"
        )

        assert X.is_complex()

    def test_circular_padding(self):
        """Test circular padding mode."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(
            x, window=window, padding_mode="circular"
        )

        assert X.is_complex()

    def test_center_false(self):
        """Test with center=False (no centering padding)."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X_centered = short_time_fourier_transform(
            x, window=window, center=True
        )
        X_not_centered = short_time_fourier_transform(
            x, window=window, center=False
        )

        # Not centered should have fewer frames
        assert X_not_centered.shape[-1] < X_centered.shape[-1]

    def test_explicit_padding(self):
        """Test explicit padding parameter."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(
            x, window=window, padding=16, center=False
        )

        assert X.is_complex()


class TestShortTimeFourierTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck_real_input(self):
        """Test gradient correctness for real input."""
        x = torch.randn(64, dtype=torch.float64, requires_grad=True)
        window = torch.hann_window(16, dtype=torch.float64)

        def stft_wrapper(x):
            return short_time_fourier_transform(x, window=window, center=False)

        assert gradcheck(stft_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_gradgradcheck_real_input(self):
        """Test second-order gradient correctness for real input."""
        x = torch.randn(32, dtype=torch.float64, requires_grad=True)
        window = torch.hann_window(16, dtype=torch.float64)

        def stft_wrapper(x):
            return short_time_fourier_transform(x, window=window, center=False)

        assert gradgradcheck(
            stft_wrapper, (x,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        x = torch.randn(128, requires_grad=True)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window)

        loss = X.abs().sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestShortTimeFourierTransformMeta:
    """Tests for meta tensor support (shape inference)."""

    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        x = torch.randn(256, device="meta")
        window = torch.hann_window(64, device="meta")
        X = short_time_fourier_transform(x, window=window)

        assert X.device == torch.device("meta")
        assert X.shape[0] == 33  # n_fft//2 + 1

    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        x = torch.randn(4, 256, device="meta")
        window = torch.hann_window(64, device="meta")
        X = short_time_fourier_transform(x, window=window, dim=-1)

        assert X.device == torch.device("meta")
        assert X.shape[0] == 4  # batch preserved
        assert X.shape[1] == 33  # freq bins


class TestShortTimeFourierTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(256, device="cuda")
        window = torch.hann_window(64, device="cuda")
        X = short_time_fourier_transform(x, window=window)

        assert X.device.type == "cuda"
        assert X.is_complex()


class TestShortTimeFourierTransformParameterOrder:
    """Tests for parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        # This should work
        X = short_time_fourier_transform(
            x, window=window, hop_length=16, n_fft=32, norm="ortho"
        )
        assert X.is_complex()

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        # This should fail - window should be keyword only
        with pytest.raises(TypeError):
            short_time_fourier_transform(x, window)  # type: ignore


class TestShortTimeFourierTransformMatchesTorchSTFT:
    """Tests that verify output matches torch.stft."""

    def test_matches_torch_stft_basic(self):
        """Test that output matches torch.stft for basic input."""
        x = torch.randn(256)
        window = torch.hann_window(64)

        X = short_time_fourier_transform(
            x, window=window, center=True, padding_mode="reflect"
        )
        expected = torch.stft(
            x,
            n_fft=64,
            hop_length=16,
            win_length=64,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

        assert torch.allclose(X, expected, atol=1e-5)

    def test_matches_torch_stft_no_center(self):
        """Test that output matches torch.stft with center=False."""
        x = torch.randn(256)
        window = torch.hann_window(64)

        X = short_time_fourier_transform(x, window=window, center=False)
        expected = torch.stft(
            x,
            n_fft=64,
            hop_length=16,
            win_length=64,
            window=window,
            center=False,
            return_complex=True,
        )

        assert torch.allclose(X, expected, atol=1e-5)


class TestShortTimeFourierTransformDim:
    """Tests for dim parameter handling.

    STFT transforms one dimension into two (freq, frames).
    The output shape is always (..., freq, frames) where ... are the
    remaining batch dimensions (with the signal dim removed).
    """

    def test_dim_middle(self):
        """Test STFT on middle dimension."""
        x = torch.randn(4, 256, 3)
        window = torch.hann_window(64)
        X = short_time_fourier_transform(x, window=window, dim=1)

        # Signal dim (1) is transformed to (freq, frames) at the end
        # Remaining dims: (4, 3) -> output: (4, 3, freq, frames)
        assert X.shape[0] == 4  # first batch dim preserved
        assert X.shape[1] == 3  # second batch dim (was after signal dim)
        assert X.shape[2] == 33  # freq bins = n_fft//2 + 1

    def test_dim_first(self):
        """Test STFT on first dimension."""
        x = torch.randn(256, 4, 3)
        window = torch.hann_window(64)
        X = short_time_fourier_transform(x, window=window, dim=0)

        # Signal dim (0) is transformed to (freq, frames) at the end
        # Remaining dims: (4, 3) -> output: (4, 3, freq, frames)
        assert X.shape[0] == 4
        assert X.shape[1] == 3
        assert X.shape[2] == 33  # freq bins

    def test_negative_dim(self):
        """Test STFT with negative dimension."""
        x = torch.randn(4, 256)
        window = torch.hann_window(64)
        X = short_time_fourier_transform(x, window=window, dim=-1)

        # Output: (4, freq, frames)
        assert X.shape[0] == 4  # batch preserved
        assert X.shape[1] == 33  # freq bins


class TestShortTimeFourierTransformNParameter:
    """Tests for n parameter (signal length control)."""

    def test_n_larger_than_input(self):
        """Test with n larger than input size."""
        x = torch.randn(64)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window, n=128)

        # Should have more frames due to longer signal
        assert X.is_complex()

    def test_n_smaller_than_input(self):
        """Test with n smaller than input size."""
        x = torch.randn(256)
        window = torch.hann_window(32)
        X = short_time_fourier_transform(x, window=window, n=64)

        # Should have fewer frames due to truncation
        assert X.is_complex()


class TestShortTimeFourierTransformVmap:
    """Tests for STFT with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        x = torch.randn(8, 256, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        # Manual batching
        X_batched = short_time_fourier_transform(x, window=window, dim=-1)

        # vmap
        def stft_single(xi):
            return short_time_fourier_transform(xi, window=window)

        X_vmap = torch.vmap(stft_single)(x)

        assert torch.allclose(X_batched, X_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        x = torch.randn(4, 4, 128, dtype=torch.float64)
        window = torch.hann_window(32, dtype=torch.float64)

        def stft_single(xi):
            return short_time_fourier_transform(xi, window=window)

        X_vmap = torch.vmap(torch.vmap(stft_single))(x)

        assert X_vmap.ndim == 4  # batch, batch, freq, frames


class TestShortTimeFourierTransformCompile:
    """Tests for STFT with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        x = torch.randn(256, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_stft(xi):
            return short_time_fourier_transform(xi, window=window)

        X_compiled = compiled_stft(x)
        X_eager = short_time_fourier_transform(x, window=window)

        assert torch.allclose(X_compiled, X_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        x = torch.randn(128, dtype=torch.float64, requires_grad=True)
        window = torch.hann_window(32, dtype=torch.float64)

        @torch.compile(fullgraph=True)
        def compiled_stft(xi):
            return short_time_fourier_transform(
                xi, window=window, center=False
            )

        X = compiled_stft(x)
        X.abs().sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestShortTimeFourierTransformEdgeCases:
    """Test edge cases and error handling."""

    def test_zeros_input(self):
        """STFT of zeros should return zeros."""
        x = torch.zeros(256, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)
        X = short_time_fourier_transform(x, window=window, center=False)
        assert torch.allclose(X, torch.zeros_like(X))

    def test_constant_input(self):
        """STFT of constant should have energy at DC."""
        x = torch.ones(256, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)
        X = short_time_fourier_transform(x, window=window, center=False)
        # DC component (index 0) should dominate in each frame
        assert X[0, :].abs().mean() > X[1:, :].abs().mean() * 10

    def test_minimum_length_input(self):
        """STFT should work with minimum valid input length."""
        window = torch.hann_window(32, dtype=torch.float64)
        x = torch.randn(32, dtype=torch.float64)
        X = short_time_fourier_transform(x, window=window, center=False)
        assert X.ndim == 2

    def test_single_frame(self):
        """STFT with hop_length >= signal length should give single frame."""
        window = torch.hann_window(64, dtype=torch.float64)
        x = torch.randn(64, dtype=torch.float64)
        X = short_time_fourier_transform(
            x, window=window, hop_length=64, center=False
        )
        assert X.shape[-1] == 1  # Single time frame
