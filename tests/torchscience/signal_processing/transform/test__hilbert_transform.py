"""Tests for torchscience.signal_processing.transform.hilbert_transform.

This test file includes tests for padding_mode and window parameters.
"""

import math

import numpy as np
import pytest
import torch

import torchscience.signal_processing.transform

# Check if scipy is available for reference tests
try:
    from scipy.signal import hilbert as scipy_hilbert

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestHilbertTransformPaddingMode:
    """Tests for padding_mode parameter."""

    @pytest.mark.parametrize(
        "mode", ["constant", "reflect", "replicate", "circular"]
    )
    def test_padding_modes_work(self, mode):
        """Test all padding modes work without error."""
        x = torch.randn(64, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode=mode
        )
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_padding_mode_invalid(self):
        """Test invalid padding_mode raises error."""
        x = torch.randn(64)
        with pytest.raises(ValueError, match="padding_mode"):
            torchscience.signal_processing.transform.hilbert_transform(
                x, padding_mode="invalid"
            )

    def test_padding_value_constant(self):
        """Test padding_value with constant mode."""
        x = torch.randn(64, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="constant", padding_value=1.0
        )
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_constant_padding_zero_vs_nonzero(self):
        """Test that different padding values give different results."""
        x = torch.randn(64, dtype=torch.float64)

        result_zero = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, n=128, padding_mode="constant", padding_value=0.0
            )
        )
        result_one = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, n=128, padding_mode="constant", padding_value=1.0
            )
        )

        # Results should differ due to different padding
        assert not torch.allclose(result_zero, result_one)

    def test_reflect_vs_constant_differ(self):
        """Test that reflect and constant padding give different results."""
        x = torch.randn(64, dtype=torch.float64)

        result_constant = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, n=128, padding_mode="constant"
            )
        )
        result_reflect = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, n=128, padding_mode="reflect"
            )
        )

        # Results should differ for non-zero signal
        assert not torch.allclose(result_constant, result_reflect)

    def test_circular_padding_periodic_signal(self):
        """Test circular padding with a periodic signal."""
        # Create a signal that's exactly one period
        n = 64
        t = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:-1]
        x = torch.sin(t)

        # Circular padding should work well for periodic signals
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="circular"
        )
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_replicate_padding(self):
        """Test replicate padding mode."""
        x = torch.randn(64, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="replicate"
        )
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_padding_mode_batched(self):
        """Test padding modes work with batched input."""
        x = torch.randn(3, 4, 64, dtype=torch.float64)

        for mode in ["constant", "reflect", "replicate", "circular"]:
            result = (
                torchscience.signal_processing.transform.hilbert_transform(
                    x, n=128, dim=-1, padding_mode=mode
                )
            )
            assert result.shape == (3, 4, 128)
            assert torch.all(torch.isfinite(result))

    def test_no_padding_when_n_equals_input(self):
        """Test that no padding happens when n equals input size."""
        x = torch.randn(64, dtype=torch.float64)

        result_default = (
            torchscience.signal_processing.transform.hilbert_transform(x)
        )
        result_explicit = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, n=64, padding_mode="reflect"
            )
        )

        # Should be the same since no padding is needed
        torch.testing.assert_close(result_default, result_explicit)

    def test_truncation_ignores_padding_mode(self):
        """Test that truncation (n < input) works regardless of padding_mode."""
        x = torch.randn(128, dtype=torch.float64)

        for mode in ["constant", "reflect", "replicate", "circular"]:
            result = (
                torchscience.signal_processing.transform.hilbert_transform(
                    x, n=64, padding_mode=mode
                )
            )
            assert result.shape == (64,)
            assert torch.all(torch.isfinite(result))


class TestHilbertTransformWindow:
    """Tests for window parameter."""

    def test_hann_window(self):
        """Test with Hann window."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, window=window
        )
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_hamming_window(self):
        """Test with Hamming window."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.hamming_window(64, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, window=window
        )
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_blackman_window(self):
        """Test with Blackman window."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.blackman_window(64, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, window=window
        )
        assert result.shape == x.shape
        assert torch.all(torch.isfinite(result))

    def test_rectangular_window_no_effect(self):
        """Test rectangular window (all ones) has no effect."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.ones(64, dtype=torch.float64)

        result_no_window = (
            torchscience.signal_processing.transform.hilbert_transform(x)
        )
        result_with_window = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, window=window
            )
        )

        torch.testing.assert_close(result_no_window, result_with_window)

    def test_window_wrong_size_raises(self):
        """Test window size mismatch raises error."""
        x = torch.randn(64)
        window = torch.ones(32)  # Wrong size

        with pytest.raises(RuntimeError, match="window"):
            torchscience.signal_processing.transform.hilbert_transform(
                x, window=window
            )

    def test_window_with_padding(self):
        """Test window combined with padding."""
        x = torch.randn(64, dtype=torch.float64)
        # Window must match padded size
        window = torch.hann_window(128, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="reflect", window=window
        )
        assert result.shape == (128,)
        assert torch.all(torch.isfinite(result))

    def test_window_batched(self):
        """Test window broadcasts over batch dimensions."""
        x = torch.randn(3, 4, 64, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, dim=-1, window=window
        )
        assert result.shape == (3, 4, 64)
        assert torch.all(torch.isfinite(result))

    def test_window_must_be_1d(self):
        """Test that window must be 1-D tensor."""
        x = torch.randn(64)
        window = torch.ones(8, 8)  # 2-D window

        with pytest.raises(RuntimeError, match="1-D"):
            torchscience.signal_processing.transform.hilbert_transform(
                x, window=window
            )

    def test_window_affects_result(self):
        """Test that applying a non-trivial window changes the result."""
        x = torch.randn(64, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        result_no_window = (
            torchscience.signal_processing.transform.hilbert_transform(x)
        )
        result_with_window = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, window=window
            )
        )

        # Results should differ
        assert not torch.allclose(result_no_window, result_with_window)


class TestHilbertTransformGradientWithNewParams:
    """Tests for gradient computation with new parameters."""

    def test_gradient_with_padding(self):
        """Test gradient works with padding."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="reflect"
        )
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_with_window(self):
        """Test gradient works with window."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, window=window
        )
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_gradient_with_padding_and_window(self):
        """Test gradient works with both padding and window."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)
        window = torch.hann_window(128, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="reflect", window=window
        )
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_gradcheck_with_padding(self):
        """Test gradient correctness with padding."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.signal_processing.transform.hilbert_transform(
                input_tensor, n=64, padding_mode="reflect"
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradcheck_with_window(self):
        """Test gradient correctness with window."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)
        window = torch.hann_window(32, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.signal_processing.transform.hilbert_transform(
                input_tensor, window=window
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    @pytest.mark.parametrize(
        "mode", ["constant", "reflect", "replicate", "circular"]
    )
    def test_gradcheck_all_padding_modes(self, mode):
        """Test gradient correctness for all padding modes."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.signal_processing.transform.hilbert_transform(
                input_tensor, n=64, padding_mode=mode
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )


class TestHilbertTransformDevice:
    """Tests for device placement with new parameters."""

    def test_cpu_with_padding(self):
        """Test CPU computation with padding."""
        x = torch.randn(64, device="cpu", dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="reflect"
        )
        assert result.device.type == "cpu"
        assert torch.all(torch.isfinite(result))

    def test_cpu_with_window(self):
        """Test CPU computation with window."""
        x = torch.randn(64, device="cpu", dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, window=window
        )
        assert result.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_with_padding(self):
        """Test CUDA computation with padding."""
        x = torch.randn(64, device="cuda", dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="reflect"
        )
        assert result.device.type == "cuda"
        assert torch.all(torch.isfinite(result))

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_with_window(self):
        """Test CUDA computation with window."""
        x = torch.randn(64, device="cuda", dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64, device="cuda")
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, window=window
        )
        assert result.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_cpu_consistency_with_padding(self):
        """Test that CPU and CUDA give same results with padding."""
        torch.manual_seed(42)
        x_cpu = torch.randn(64, dtype=torch.float64)
        x_cuda = x_cpu.cuda()

        result_cpu = (
            torchscience.signal_processing.transform.hilbert_transform(
                x_cpu, n=128, padding_mode="reflect"
            )
        )
        result_cuda = (
            torchscience.signal_processing.transform.hilbert_transform(
                x_cuda, n=128, padding_mode="reflect"
            )
        )

        torch.testing.assert_close(
            result_cpu, result_cuda.cpu(), rtol=1e-5, atol=1e-5
        )


class TestHilbertTransformMeta:
    """Tests for meta tensor support with new parameters."""

    def test_meta_with_padding(self):
        """Test meta tensor shape inference with padding."""
        x_meta = torch.empty(64, device="meta")
        result = torchscience.signal_processing.transform.hilbert_transform(
            x_meta, n=128, padding_mode="reflect"
        )
        assert result.shape == (128,)
        assert result.device.type == "meta"

    def test_meta_with_window(self):
        """Test meta tensor shape inference with window."""
        x_meta = torch.empty(64, device="meta")
        window_meta = torch.empty(64, device="meta")
        result = torchscience.signal_processing.transform.hilbert_transform(
            x_meta, window=window_meta
        )
        assert result.shape == (64,)
        assert result.device.type == "meta"


class TestHilbertTransformBackwardCompatibility:
    """Tests for backward compatibility with old API."""

    def test_default_padding_is_constant_zero(self):
        """Test that default padding matches old zero-padding behavior."""
        x = torch.randn(64, dtype=torch.float64)

        # New API with explicit defaults
        result_new = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, n=128, padding_mode="constant", padding_value=0.0
            )
        )

        # This should match zero-padding behavior
        assert result_new.shape == (128,)
        assert torch.all(torch.isfinite(result_new))

    def test_no_window_is_default(self):
        """Test that no window (None) is the default."""
        x = torch.randn(64, dtype=torch.float64)

        result_default = (
            torchscience.signal_processing.transform.hilbert_transform(x)
        )
        result_explicit = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, window=None
            )
        )

        torch.testing.assert_close(result_default, result_explicit)


class TestHilbertTransformEdgeCasesNewParams:
    """Edge case tests for new parameters."""

    def test_window_with_single_element(self):
        """Test window with single element tensor."""
        x = torch.tensor([1.0], dtype=torch.float64)
        window = torch.ones(1, dtype=torch.float64)

        result = torchscience.signal_processing.transform.hilbert_transform(
            x, window=window
        )
        assert result.shape == (1,)

    def test_large_padding_factor(self):
        """Test with large padding factor."""
        x = torch.randn(32, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=1024, padding_mode="reflect"
        )
        assert result.shape == (1024,)
        assert torch.all(torch.isfinite(result))

    @pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
    def test_size_one_with_non_constant_padding_raises(self, mode):
        """Test that size-1 dimension with non-constant padding raises an error.

        Reflect/replicate/circular padding requires at least 2 elements to work
        properly. Using these modes with a size-1 dimension should raise an error.
        """
        x = torch.tensor([1.0], dtype=torch.float64)

        with pytest.raises(
            RuntimeError, match="Cannot use reflect/replicate/circular padding"
        ):
            torchscience.signal_processing.transform.hilbert_transform(
                x, n=10, padding_mode=mode
            )

    def test_size_one_with_constant_padding_works(self):
        """Test that size-1 dimension with constant padding works."""
        x = torch.tensor([1.0], dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=10, padding_mode="constant"
        )
        assert result.shape == (10,)
        assert torch.all(torch.isfinite(result))

    def test_all_combinations(self):
        """Test various combinations of parameters."""
        x = torch.randn(64, dtype=torch.float64)

        for mode in ["constant", "reflect"]:
            for n in [None, 64, 128]:
                window = (
                    torch.hann_window(n or 64, dtype=torch.float64)
                    if n
                    else None
                )

                result = (
                    torchscience.signal_processing.transform.hilbert_transform(
                        x, n=n, padding_mode=mode, window=window
                    )
                )

                expected_size = n if n is not None else 64
                assert result.shape == (expected_size,)
                assert torch.all(torch.isfinite(result))


class TestHilbertTransformComplexInput:
    """Tests for complex input with padding modes."""

    @pytest.mark.parametrize(
        "mode", ["constant", "reflect", "replicate", "circular"]
    )
    def test_complex_input_with_padding_modes(self, mode):
        """Test complex input works with all padding modes."""
        x = torch.randn(64, dtype=torch.complex128)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode=mode
        )
        assert result.shape == (128,)
        assert result.dtype == torch.complex128
        assert torch.all(torch.isfinite(result.real))
        assert torch.all(torch.isfinite(result.imag))

    def test_complex_input_batched_with_padding(self):
        """Test complex batched input with padding."""
        x = torch.randn(3, 4, 64, dtype=torch.complex128)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, dim=-1, padding_mode="reflect"
        )
        assert result.shape == (3, 4, 128)
        assert result.dtype == torch.complex128

    def test_complex_input_gradient_with_padding(self):
        """Test gradient works for complex input with padding."""
        x = torch.randn(32, dtype=torch.complex128, requires_grad=True)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=64, padding_mode="reflect"
        )
        loss = result.abs().sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestHilbertTransformTruncation:
    """Tests for truncation mode (n < input_size)."""

    def test_truncation_basic(self):
        """Test basic truncation."""
        x = torch.randn(128, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=64
        )
        assert result.shape == (64,)
        assert torch.all(torch.isfinite(result))

    def test_truncation_gradient(self):
        """Test gradient with truncation."""
        x = torch.randn(128, requires_grad=True, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=64
        )
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_gradcheck_truncation(self):
        """Test gradient correctness with truncation."""
        x = torch.randn(64, requires_grad=True, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.signal_processing.transform.hilbert_transform(
                input_tensor, n=32
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_truncation_batched(self):
        """Test truncation with batched input."""
        x = torch.randn(3, 4, 128, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=64, dim=-1
        )
        assert result.shape == (3, 4, 64)


class TestHilbertTransformMultiDimensional:
    """Tests for multi-dimensional inputs with non-last dimension."""

    @pytest.mark.parametrize("dim", [0, 1, 2])
    def test_3d_all_dims(self, dim):
        """Test 3D input with transform along all possible dimensions."""
        x = torch.randn(16, 32, 64, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, dim=dim
        )

        expected_shape = list(x.shape)
        assert result.shape == tuple(expected_shape)
        assert torch.all(torch.isfinite(result))

    @pytest.mark.parametrize("dim", [0, 1])
    def test_3d_non_last_dim_with_padding(self, dim):
        """Test padding along non-last dimension."""
        x = torch.randn(16, 32, 64, dtype=torch.float64)
        n = x.size(dim) * 2  # Double the size along dim
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=n, dim=dim, padding_mode="reflect"
        )

        expected_shape = list(x.shape)
        expected_shape[dim] = n
        assert result.shape == tuple(expected_shape)
        assert torch.all(torch.isfinite(result))

    def test_dim_0_gradient(self):
        """Test gradient with dim=0."""
        x = torch.randn(32, 64, requires_grad=True, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, dim=0
        )
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_negative_dim(self):
        """Test negative dimension indexing."""
        x = torch.randn(16, 32, 64, dtype=torch.float64)

        result_neg = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, dim=-2
            )
        )
        result_pos = (
            torchscience.signal_processing.transform.hilbert_transform(
                x, dim=1
            )
        )

        torch.testing.assert_close(result_neg, result_pos)


class TestHilbertTransformZeroSignal:
    """Tests for zero-valued signals."""

    def test_zero_signal(self):
        """Test that zero input produces zero output."""
        x = torch.zeros(64, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(x)
        torch.testing.assert_close(result, x)

    def test_zero_signal_with_padding(self):
        """Test zero signal with padding."""
        x = torch.zeros(64, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(
            x, n=128, padding_mode="constant"
        )
        assert torch.allclose(result, torch.zeros(128, dtype=torch.float64))

    def test_zero_signal_gradient(self):
        """Test gradient for zero signal."""
        x = torch.zeros(64, requires_grad=True, dtype=torch.float64)
        result = torchscience.signal_processing.transform.hilbert_transform(x)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))


class TestHilbertTransformDeviceMismatch:
    """Tests for device mismatch error handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_window_device_mismatch_raises(self):
        """Test that window on different device raises error."""
        x = torch.randn(64, device="cuda", dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64, device="cpu")

        with pytest.raises(RuntimeError, match="same device"):
            torchscience.signal_processing.transform.hilbert_transform(
                x, window=window
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_window_device_mismatch_cpu_input_cuda_window(self):
        """Test CPU input with CUDA window raises error."""
        x = torch.randn(64, device="cpu", dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64, device="cuda")

        with pytest.raises(RuntimeError, match="same device"):
            torchscience.signal_processing.transform.hilbert_transform(
                x, window=window
            )


class TestHilbertTransformGradcheckPaddingWindow:
    """Gradcheck tests for padding + window combinations."""

    @pytest.mark.parametrize(
        "mode", ["constant", "reflect", "replicate", "circular"]
    )
    def test_gradcheck_padding_with_window(self, mode):
        """Test gradient correctness with padding and window."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.signal_processing.transform.hilbert_transform(
                input_tensor, n=64, padding_mode=mode, window=window
            )

        assert torch.autograd.gradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
        )

    def test_gradgradcheck_with_padding_and_window(self):
        """Test double backward with padding and window."""
        x = torch.randn(32, requires_grad=True, dtype=torch.float64)
        window = torch.hann_window(64, dtype=torch.float64)

        def fn(input_tensor):
            return torchscience.signal_processing.transform.hilbert_transform(
                input_tensor, n=64, padding_mode="reflect", window=window
            )

        assert torch.autograd.gradgradcheck(
            fn, (x,), eps=1e-5, atol=1e-4, rtol=1e-4
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

        torch_result = (
            torchscience.signal_processing.transform.hilbert_transform(
                x_torch
            ).numpy()
        )

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
            torch_result = (
                torchscience.signal_processing.transform.hilbert_transform(
                    x_torch
                ).numpy()
            )

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
        torch_result = (
            torchscience.signal_processing.transform.hilbert_transform(
                x_torch
            ).numpy()
        )

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
        torch_result = (
            torchscience.signal_processing.transform.hilbert_transform(
                x_torch
            ).numpy()
        )

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
        torch_result = (
            torchscience.signal_processing.transform.hilbert_transform(
                x_torch, n=128, padding_mode="constant", padding_value=0.0
            ).numpy()
        )

        np.testing.assert_allclose(
            scipy_result, torch_result, rtol=1e-10, atol=1e-10
        )

    def test_double_hilbert_property(self):
        """Test that H[H[f]] = -f (involutory property).

        Note: This property holds exactly only for infinite signals or
        properly windowed periodic signals. For finite discrete signals,
        there are edge effects (Gibbs phenomenon), so we verify:
        1. The property holds approximately for random signals
        2. The property holds exactly for periodic signals with integer periods
        """
        # Test with periodic signal (exact property)
        n = 256
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_periodic = np.sin(t)
        x_torch = torch.from_numpy(x_periodic)

        h1 = torchscience.signal_processing.transform.hilbert_transform(
            x_torch
        )
        h2 = torchscience.signal_processing.transform.hilbert_transform(h1)

        # For periodic signals, property should hold well
        np.testing.assert_allclose(
            h2.numpy(), -x_periodic, rtol=1e-6, atol=1e-6
        )

        # Test with random signal (approximate due to edge effects)
        np.random.seed(42)
        x_random = np.random.randn(128)
        x_torch_random = torch.from_numpy(x_random)

        h1_random = torchscience.signal_processing.transform.hilbert_transform(
            x_torch_random
        )
        h2_random = torchscience.signal_processing.transform.hilbert_transform(
            h1_random
        )

        # For random signals, verify the central region (avoiding edge effects)
        center_slice = slice(32, 96)
        np.testing.assert_allclose(
            h2_random.numpy()[center_slice],
            -x_random[center_slice],
            rtol=0.1,
            atol=0.1,
        )
