"""Tests for inverse_short_time_fourier_transform."""

import pytest
import torch
from torch.autograd import gradcheck

from torchscience.transform import (
    inverse_short_time_fourier_transform,
    short_time_fourier_transform,
)


class TestInverseShortTimeFourierTransformBasic:
    """Tests for basic ISTFT functionality."""

    def test_roundtrip(self):
        """Test that STFT -> ISTFT roundtrip reconstructs the signal."""
        x = torch.randn(1024)
        window = torch.hann_window(256)
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=1024
        )

        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_output_is_real(self):
        """ISTFT output should be real for real input STFT."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        # Output should be real (not complex)
        assert not x_rec.is_complex()

    def test_batched_roundtrip(self):
        """Test roundtrip with batched input."""
        x = torch.randn(4, 512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window, dim=-1)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert x_rec.shape == x.shape
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_multi_batch_dims_roundtrip(self):
        """Test roundtrip with multiple batch dimensions."""
        x = torch.randn(2, 3, 256)
        window = torch.hann_window(64)
        S = short_time_fourier_transform(x, window=window, dim=-1)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=256
        )

        assert x_rec.shape == x.shape
        assert torch.allclose(x, x_rec, atol=1e-5)


class TestInverseShortTimeFourierTransformNormalization:
    """Tests for normalization modes."""

    def test_roundtrip_backward_norm(self):
        """Test roundtrip with backward normalization."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window, norm="backward")
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512, norm="backward"
        )

        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_roundtrip_ortho_norm(self):
        """Test roundtrip with ortho normalization."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window, norm="ortho")
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512, norm="ortho"
        )

        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_roundtrip_forward_norm(self):
        """Test roundtrip with forward normalization."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window, norm="forward")
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512, norm="forward"
        )

        assert torch.allclose(x, x_rec, atol=1e-5)


class TestInverseShortTimeFourierTransformWindow:
    """Tests for window handling."""

    def test_window_required(self):
        """Test that window parameter is required."""
        x = torch.randn(128)
        window = torch.hann_window(32)
        S = short_time_fourier_transform(x, window=window)

        with pytest.raises(ValueError, match="window"):
            inverse_short_time_fourier_transform(S, window=None)

    def test_hann_window_roundtrip(self):
        """Test roundtrip with Hann window."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_hamming_window_roundtrip(self):
        """Test roundtrip with Hamming window."""
        x = torch.randn(512)
        window = torch.hamming_window(128)
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_window_must_be_1d(self):
        """Test that window must be 1-D."""
        S = torch.randn(33, 10, dtype=torch.complex64)
        window = torch.ones(4, 8)  # 2D window

        with pytest.raises(ValueError, match="1-D"):
            inverse_short_time_fourier_transform(S, window=window)


class TestInverseShortTimeFourierTransformLengthParameter:
    """Tests for length/n parameter handling."""

    def test_length_parameter(self):
        """Test explicit length parameter."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert x_rec.shape[-1] == 512

    def test_n_parameter_alias(self):
        """Test that n is an alias for length."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(S, window=window, n=512)

        assert x_rec.shape[-1] == 512
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_length_takes_precedence_over_n(self):
        """Test that length takes precedence over n when both provided."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512, n=256
        )

        # length should win
        assert x_rec.shape[-1] == 512


class TestInverseShortTimeFourierTransformNfft:
    """Tests for n_fft parameter handling."""

    def test_default_n_fft_inferred(self):
        """Test that n_fft is inferred from input shape."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        # n_fft defaults to window size = 128, so freq bins = 65
        S = short_time_fourier_transform(x, window=window)
        assert S.shape[-2] == 65  # 128//2 + 1

        # ISTFT should infer n_fft = 2 * (65 - 1) = 128
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )
        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_explicit_n_fft(self):
        """Test explicit n_fft parameter."""
        x = torch.randn(512)
        window = torch.hann_window(64)
        S = short_time_fourier_transform(x, window=window, n_fft=128)

        x_rec = inverse_short_time_fourier_transform(
            S, window=window, n_fft=128, length=512
        )
        assert torch.allclose(x, x_rec, atol=1e-5)


class TestInverseShortTimeFourierTransformHopLength:
    """Tests for hop_length parameter handling."""

    def test_default_hop_length(self):
        """Test default hop_length = window.size(0) // 4."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        # Default hop_length = 128 // 4 = 32
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_custom_hop_length(self):
        """Test custom hop_length."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        hop_length = 64
        S = short_time_fourier_transform(
            x, window=window, hop_length=hop_length
        )
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, hop_length=hop_length, length=512
        )

        assert torch.allclose(x, x_rec, atol=1e-5)


class TestInverseShortTimeFourierTransformGradient:
    """Tests for gradient computation."""

    def test_gradcheck(self):
        """Test gradient correctness."""
        x = torch.randn(128, dtype=torch.float64)
        window = torch.hann_window(32, dtype=torch.float64)
        # Use center=True for gradcheck (Hann window with center=False
        # doesn't satisfy COLA properly for boundary frames)
        S = short_time_fourier_transform(x, window=window, center=True)
        S_input = S.detach().requires_grad_(True)

        def istft_wrapper(s):
            return inverse_short_time_fourier_transform(
                s, window=window, length=128, center=True
            )

        assert gradcheck(
            istft_wrapper, (S_input,), eps=1e-6, atol=1e-4, rtol=1e-3
        )

    def test_backward_pass(self):
        """Test that backward pass works."""
        x = torch.randn(256)
        window = torch.hann_window(64)
        S = short_time_fourier_transform(x, window=window)
        S_detached = S.detach().requires_grad_(True)

        x_rec = inverse_short_time_fourier_transform(
            S_detached, window=window, length=256
        )
        loss = x_rec.sum()
        loss.backward()

        assert S_detached.grad is not None
        assert S_detached.grad.shape == S_detached.shape


class TestInverseShortTimeFourierTransformParameterOrder:
    """Tests for parameter ordering (keyword-only)."""

    def test_all_parameters_keyword_only(self):
        """Test that all parameters after input are keyword-only."""
        S = torch.randn(65, 10, dtype=torch.complex64)
        window = torch.hann_window(128)

        # This should work
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, hop_length=32, n_fft=128, norm="ortho"
        )
        assert not x_rec.is_complex()

    def test_positional_input_only(self):
        """Test that only input can be positional."""
        S = torch.randn(65, 10, dtype=torch.complex64)
        window = torch.hann_window(128)

        # This should fail - window should be keyword only
        with pytest.raises(TypeError):
            inverse_short_time_fourier_transform(S, window)  # type: ignore


class TestInverseShortTimeFourierTransformCenter:
    """Tests for center parameter."""

    def test_center_true_roundtrip(self):
        """Test roundtrip with center=True (default)."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        S = short_time_fourier_transform(x, window=window, center=True)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512, center=True
        )

        assert torch.allclose(x, x_rec, atol=1e-5)

    def test_center_false_roundtrip(self):
        """Test roundtrip with center=False.

        Note: center=False with tapered windows (Hann, Hamming) doesn't satisfy
        the COLA condition at boundaries. Use a rectangular window instead.
        """
        x = torch.randn(512)
        # Rectangular window satisfies COLA with center=False
        window = torch.ones(128)
        S = short_time_fourier_transform(x, window=window, center=False)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512, center=False
        )

        assert torch.allclose(x, x_rec, atol=1e-5)


class TestInverseShortTimeFourierTransformMeta:
    """Tests for meta tensor support (shape inference).

    Note: torch.istft does not currently support meta tensors due to
    missing fake/meta kernel implementations in PyTorch. These tests
    are skipped until upstream support is added.
    """

    @pytest.mark.skip(reason="torch.istft does not support meta tensors")
    def test_meta_tensor_shape(self):
        """Test shape inference with meta tensors."""
        S = torch.randn(65, 20, dtype=torch.complex64, device="meta")
        window = torch.hann_window(128, device="meta")
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert x_rec.device == torch.device("meta")
        assert x_rec.shape[-1] == 512

    @pytest.mark.skip(reason="torch.istft does not support meta tensors")
    def test_meta_tensor_batched(self):
        """Test shape inference with batched meta tensors."""
        S = torch.randn(4, 65, 20, dtype=torch.complex64, device="meta")
        window = torch.hann_window(128, device="meta")
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert x_rec.device == torch.device("meta")
        assert x_rec.shape == (4, 512)


class TestInverseShortTimeFourierTransformDevice:
    """Tests for device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_tensor(self):
        """Test that CUDA tensors work."""
        x = torch.randn(512, device="cuda")
        window = torch.hann_window(128, device="cuda")
        S = short_time_fourier_transform(x, window=window)
        x_rec = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert x_rec.device.type == "cuda"
        assert not x_rec.is_complex()


class TestInverseShortTimeFourierTransformMatchesTorchISTFT:
    """Tests that verify output matches torch.istft."""

    def test_matches_torch_istft_basic(self):
        """Test that output matches torch.istft for basic input."""
        x = torch.randn(512)
        window = torch.hann_window(128)
        n_fft = 128
        hop_length = 32

        S = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=128,
            window=window,
            center=True,
            return_complex=True,
        )

        x_rec = inverse_short_time_fourier_transform(
            S,
            window=window,
            n_fft=n_fft,
            hop_length=hop_length,
            length=512,
            center=True,
        )

        expected = torch.istft(
            S,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=128,
            window=window,
            center=True,
            length=512,
        )

        assert torch.allclose(x_rec, expected, atol=1e-5)


class TestInverseShortTimeFourierTransformVmap:
    """Tests for ISTFT with vmap."""

    def test_vmap_basic(self):
        """vmap should batch over first dimension."""
        x = torch.randn(8, 512, dtype=torch.float64)
        window = torch.hann_window(128, dtype=torch.float64)

        # Get STFT
        S = short_time_fourier_transform(x, window=window, dim=-1)

        # Manual batching - ISTFT on batched input
        x_rec_batched = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        # vmap
        def istft_single(Si):
            return inverse_short_time_fourier_transform(
                Si, window=window, length=512
            )

        x_rec_vmap = torch.vmap(istft_single)(S)

        assert torch.allclose(x_rec_batched, x_rec_vmap, atol=1e-10)

    def test_vmap_nested(self):
        """Nested vmap should work."""
        x = torch.randn(4, 4, 256, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        S = short_time_fourier_transform(x, window=window, dim=-1)

        def istft_single(Si):
            return inverse_short_time_fourier_transform(
                Si, window=window, length=256
            )

        x_rec_vmap = torch.vmap(torch.vmap(istft_single))(S)

        assert x_rec_vmap.shape == x.shape


class TestInverseShortTimeFourierTransformCompile:
    """Tests for ISTFT with torch.compile."""

    def test_compile_basic(self):
        """torch.compile should work."""
        x = torch.randn(512, dtype=torch.float64)
        window = torch.hann_window(128, dtype=torch.float64)
        S = short_time_fourier_transform(x, window=window)

        @torch.compile(fullgraph=True)
        def compiled_istft(Si):
            return inverse_short_time_fourier_transform(
                Si, window=window, length=512
            )

        x_rec_compiled = compiled_istft(S)
        x_rec_eager = inverse_short_time_fourier_transform(
            S, window=window, length=512
        )

        assert torch.allclose(x_rec_compiled, x_rec_eager, atol=1e-10)

    def test_compile_with_grad(self):
        """torch.compile should work with gradients."""
        x = torch.randn(256, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)
        S = short_time_fourier_transform(x, window=window)
        S_input = S.detach().requires_grad_(True)

        @torch.compile(fullgraph=True)
        def compiled_istft(Si):
            return inverse_short_time_fourier_transform(
                Si, window=window, length=256
            )

        x_rec = compiled_istft(S_input)
        x_rec.sum().backward()

        assert S_input.grad is not None
        assert S_input.grad.shape == S_input.shape
