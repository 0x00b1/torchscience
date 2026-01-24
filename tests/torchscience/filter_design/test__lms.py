"""Tests for LMS (Least Mean Squares) adaptive filter."""

import torch

from torchscience.filter_design import lms


class TestLMSConverges:
    """Test that LMS converges on system identification problems."""

    def test_lms_converges_simple_system(self) -> None:
        """Test LMS converges to identify a simple FIR system."""
        torch.manual_seed(42)

        # True system (FIR filter)
        h_true = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate input signal
        n_samples = 1000
        x = torch.randn(n_samples, dtype=torch.float64)

        # Generate desired signal (output of true system)
        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run LMS
        y, w = lms(x, d, num_taps=num_taps, mu=0.1, return_weights=True)

        # Final weights should be close to true system
        torch.testing.assert_close(w, h_true, rtol=0.1, atol=0.05)

    def test_lms_converges_with_small_step_size(self) -> None:
        """Test LMS converges with small step size (slower but more accurate)."""
        torch.manual_seed(123)

        # True system
        h_true = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate input signal
        n_samples = 5000  # More samples for smaller step size
        x = torch.randn(n_samples, dtype=torch.float64)

        # Generate desired signal
        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run LMS with small step size
        y, w = lms(x, d, num_taps=num_taps, mu=0.01, return_weights=True)

        # Final weights should be close to true system
        torch.testing.assert_close(w, h_true, rtol=0.05, atol=0.02)

    def test_lms_output_reduces_error(self) -> None:
        """Test that LMS output error decreases over time."""
        torch.manual_seed(999)

        # True system
        h_true = torch.tensor([0.8, -0.4], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate signals
        n_samples = 500
        x = torch.randn(n_samples, dtype=torch.float64)

        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run LMS
        y = lms(x, d, num_taps=num_taps, mu=0.1)

        # Compute error in first and second half
        e = d - y
        e_first_half = e[: n_samples // 2].pow(2).mean()
        e_second_half = e[n_samples // 2 :].pow(2).mean()

        # Second half should have smaller error (converged more)
        assert e_second_half < e_first_half


class TestLMSNoiseCancellation:
    """Test LMS for adaptive noise cancellation."""

    def test_lms_noise_cancellation(self) -> None:
        """Test LMS can cancel correlated noise."""
        torch.manual_seed(42)

        n_samples = 1000

        # Clean signal (sinusoid)
        t = torch.linspace(
            0, 2 * torch.pi * 10, n_samples, dtype=torch.float64
        )
        clean_signal = torch.sin(t)

        # Noise source
        noise_ref = torch.randn(n_samples, dtype=torch.float64)

        # Noise that contaminates the signal (filtered version of noise_ref)
        h_noise = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
        noise = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(len(h_noise)):
                if n - k >= 0:
                    noise[n] += h_noise[k] * noise_ref[n - k]

        # Corrupted signal (clean + noise)
        d = clean_signal + noise

        # Use LMS to estimate the noise path
        num_taps = 5
        y = lms(noise_ref, d, num_taps=num_taps, mu=0.05)

        # The estimated noise should be subtracted from d to recover clean signal
        recovered = d - y

        # In the second half (after convergence), recovered should be closer to clean
        second_half = slice(n_samples // 2, None)
        error_before = (
            (d[second_half] - clean_signal[second_half]).pow(2).mean()
        )
        error_after = (
            (recovered[second_half] - clean_signal[second_half]).pow(2).mean()
        )

        # Noise cancellation should improve SNR
        assert error_after < error_before


class TestLMSGradients:
    """Test LMS gradient computation."""

    def test_gradcheck_weights_gradients(self) -> None:
        """Gradient check for initial weights."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64)
        d = torch.randn(50, dtype=torch.float64)
        w0 = torch.randn(4, dtype=torch.float64, requires_grad=True)

        def fn(w0_):
            return lms(x, d, num_taps=4, mu=0.01, w0=w0_)

        torch.autograd.gradcheck(
            fn,
            (w0,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_input_signal(self) -> None:
        """Gradient check for input signal."""
        torch.manual_seed(42)

        x = torch.randn(30, dtype=torch.float64, requires_grad=True)
        d = torch.randn(30, dtype=torch.float64)

        def fn(x_):
            return lms(x_, d, num_taps=3, mu=0.01)

        torch.autograd.gradcheck(
            fn,
            (x,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_desired_signal(self) -> None:
        """Gradient check for desired signal."""
        torch.manual_seed(42)

        x = torch.randn(30, dtype=torch.float64)
        d = torch.randn(30, dtype=torch.float64, requires_grad=True)

        def fn(d_):
            return lms(x, d_, num_taps=3, mu=0.01)

        torch.autograd.gradcheck(
            fn,
            (d,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestLMSBatchSignals:
    """Test LMS with batched inputs."""

    def test_batch_2d(self) -> None:
        """Test LMS with 2D batched signals."""
        torch.manual_seed(42)

        batch_size = 4
        n_samples = 100
        num_taps = 3

        x = torch.randn(batch_size, n_samples, dtype=torch.float64)
        d = torch.randn(batch_size, n_samples, dtype=torch.float64)

        y = lms(x, d, num_taps=num_taps, mu=0.05)

        assert y.shape == (batch_size, n_samples)

    def test_batch_3d(self) -> None:
        """Test LMS with 3D batched signals."""
        torch.manual_seed(42)

        shape = (2, 3, 100)
        num_taps = 4

        x = torch.randn(*shape, dtype=torch.float64)
        d = torch.randn(*shape, dtype=torch.float64)

        y = lms(x, d, num_taps=num_taps, mu=0.05)

        assert y.shape == shape

    def test_batch_matches_individual(self) -> None:
        """Test that batched result matches processing individually."""
        torch.manual_seed(42)

        batch_size = 3
        n_samples = 50
        num_taps = 3
        mu = 0.05

        x = torch.randn(batch_size, n_samples, dtype=torch.float64)
        d = torch.randn(batch_size, n_samples, dtype=torch.float64)

        # Batched processing
        y_batch = lms(x, d, num_taps=num_taps, mu=mu)

        # Individual processing
        y_individual = []
        for i in range(batch_size):
            y_i = lms(x[i], d[i], num_taps=num_taps, mu=mu)
            y_individual.append(y_i)
        y_individual = torch.stack(y_individual)

        torch.testing.assert_close(
            y_batch, y_individual, rtol=1e-10, atol=1e-10
        )


class TestLMSInitialWeights:
    """Test LMS with initial weight specification."""

    def test_initial_weights_zeros(self) -> None:
        """Test that default initial weights are zeros."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 4

        # Default (zeros)
        y1 = lms(x, d, num_taps=num_taps, mu=0.05)

        # Explicit zeros
        w0 = torch.zeros(num_taps, dtype=torch.float64)
        y2 = lms(x, d, num_taps=num_taps, mu=0.05, w0=w0)

        torch.testing.assert_close(y1, y2, rtol=1e-10, atol=1e-10)

    def test_initial_weights_nonzero(self) -> None:
        """Test LMS with non-zero initial weights."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 3

        # Non-zero initial weights
        w0 = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)

        y, w = lms(
            x, d, num_taps=num_taps, mu=0.05, w0=w0, return_weights=True
        )

        assert y.shape == x.shape
        assert w.shape == w0.shape


class TestLMSReturnWeights:
    """Test LMS return_weights option."""

    def test_return_weights_false(self) -> None:
        """Test that return_weights=False returns only y."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)

        result = lms(x, d, num_taps=4, mu=0.05, return_weights=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_return_weights_true(self) -> None:
        """Test that return_weights=True returns (y, w) tuple."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 4

        result = lms(x, d, num_taps=num_taps, mu=0.05, return_weights=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        y, w = result
        assert y.shape == x.shape
        assert w.shape == (num_taps,)


class TestLMSDtypeAndDevice:
    """Test LMS dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float32)
        d = torch.randn(100, dtype=torch.float32)

        y = lms(x, d, num_taps=4, mu=0.05)

        assert y.dtype == torch.float32

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)

        y = lms(x, d, num_taps=4, mu=0.05)

        assert y.dtype == torch.float64

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64, device="cpu")
        d = torch.randn(100, dtype=torch.float64, device="cpu")

        y = lms(x, d, num_taps=4, mu=0.05)

        assert y.device == x.device


class TestLMSEdgeCases:
    """Test LMS edge cases."""

    def test_single_tap(self) -> None:
        """Test LMS with single tap (adaptive gain)."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = 0.7 * x  # Desired is just a scaled version

        y, w = lms(x, d, num_taps=1, mu=0.1, return_weights=True)

        # Weight should converge to ~0.7
        assert abs(w[0].item() - 0.7) < 0.1

    def test_short_signal(self) -> None:
        """Test LMS with short signal."""
        torch.manual_seed(42)

        x = torch.randn(10, dtype=torch.float64)
        d = torch.randn(10, dtype=torch.float64)

        y = lms(x, d, num_taps=3, mu=0.05)

        assert y.shape == x.shape

    def test_num_taps_equals_signal_length(self) -> None:
        """Test when num_taps equals signal length."""
        torch.manual_seed(42)

        n_samples = 10
        x = torch.randn(n_samples, dtype=torch.float64)
        d = torch.randn(n_samples, dtype=torch.float64)

        y = lms(x, d, num_taps=n_samples, mu=0.01)

        assert y.shape == x.shape

    def test_zero_step_size(self) -> None:
        """Test that mu=0 means no weight updates."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 3

        w0 = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)

        y, w = lms(x, d, num_taps=num_taps, mu=0.0, w0=w0, return_weights=True)

        # Weights should remain unchanged
        torch.testing.assert_close(w, w0, rtol=1e-10, atol=1e-10)
