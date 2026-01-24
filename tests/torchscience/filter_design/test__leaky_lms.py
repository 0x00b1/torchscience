"""Tests for Leaky LMS adaptive filter."""

import torch

from torchscience.filter_design import leaky_lms


class TestLeakyLMSConverges:
    """Test that Leaky LMS converges on system identification problems."""

    def test_leaky_lms_converges_simple_system(self) -> None:
        """Test Leaky LMS converges to identify a simple FIR system."""
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

        # Run Leaky LMS
        y, w = leaky_lms(
            x, d, num_taps=num_taps, mu=0.1, gamma=0.001, return_weights=True
        )

        # Final weights should be close to true system
        # Note: leaky LMS may have slight bias due to leakage
        torch.testing.assert_close(w, h_true, rtol=0.15, atol=0.1)

    def test_leaky_lms_output_reduces_error(self) -> None:
        """Test that Leaky LMS output error decreases over time."""
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

        # Run Leaky LMS
        y = leaky_lms(x, d, num_taps=num_taps, mu=0.1, gamma=0.001)

        # Compute error in first and second half
        e = d - y
        e_first_half = e[: n_samples // 2].pow(2).mean()
        e_second_half = e[n_samples // 2 :].pow(2).mean()

        # Second half should have smaller error (converged more)
        assert e_second_half < e_first_half


class TestLeakyLMSLeakage:
    """Test Leaky LMS leakage behavior."""

    def test_leakage_prevents_weight_explosion(self) -> None:
        """Test that leakage prevents weights from growing unboundedly."""
        torch.manual_seed(42)

        num_taps = 4
        n_samples = 1000

        # Input with some correlation that could cause weight explosion in LMS
        x = torch.randn(n_samples, dtype=torch.float64)
        d = torch.randn(n_samples, dtype=torch.float64) * 10  # Large desired

        # Run Leaky LMS with moderate leakage
        y, w = leaky_lms(
            x, d, num_taps=num_taps, mu=0.1, gamma=0.01, return_weights=True
        )

        # Weights should remain bounded
        assert torch.abs(w).max() < 100

    def test_leakage_decays_weights_toward_zero(self) -> None:
        """Test that leakage causes weights to decay toward zero."""
        torch.manual_seed(42)

        num_taps = 3
        n_samples = 500

        # Zero desired signal - weights should decay to zero
        x = torch.randn(n_samples, dtype=torch.float64)
        d = torch.zeros(n_samples, dtype=torch.float64)

        # Start with non-zero weights
        w0 = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        y, w = leaky_lms(
            x,
            d,
            num_taps=num_taps,
            mu=0.01,
            gamma=0.1,  # Strong leakage
            w0=w0,
            return_weights=True,
        )

        # Final weights should be smaller than initial (decayed)
        assert torch.abs(w).mean() < torch.abs(w0).mean()

    def test_zero_gamma_equals_standard_lms(self) -> None:
        """Test that gamma=0 gives standard LMS behavior."""
        from torchscience.filter_design import lms

        torch.manual_seed(42)

        n_samples = 100
        num_taps = 3
        mu = 0.05

        x = torch.randn(n_samples, dtype=torch.float64)
        d = torch.randn(n_samples, dtype=torch.float64)

        # Leaky LMS with gamma=0
        y_leaky = leaky_lms(x, d, num_taps=num_taps, mu=mu, gamma=0.0)

        # Standard LMS
        y_lms = lms(x, d, num_taps=num_taps, mu=mu)

        torch.testing.assert_close(y_leaky, y_lms, rtol=1e-10, atol=1e-10)


class TestLeakyLMSGradients:
    """Test Leaky LMS gradient computation."""

    def test_gradcheck_weights_gradients(self) -> None:
        """Gradient check for initial weights."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64)
        d = torch.randn(50, dtype=torch.float64)
        w0 = torch.randn(4, dtype=torch.float64, requires_grad=True)

        def fn(w0_):
            return leaky_lms(x, d, num_taps=4, mu=0.01, gamma=0.001, w0=w0_)

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
            return leaky_lms(x_, d, num_taps=3, mu=0.01, gamma=0.001)

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
            return leaky_lms(x, d_, num_taps=3, mu=0.01, gamma=0.001)

        torch.autograd.gradcheck(
            fn,
            (d,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestLeakyLMSBatchSignals:
    """Test Leaky LMS with batched inputs."""

    def test_batch_2d(self) -> None:
        """Test Leaky LMS with 2D batched signals."""
        torch.manual_seed(42)

        batch_size = 4
        n_samples = 100
        num_taps = 3

        x = torch.randn(batch_size, n_samples, dtype=torch.float64)
        d = torch.randn(batch_size, n_samples, dtype=torch.float64)

        y = leaky_lms(x, d, num_taps=num_taps, mu=0.05, gamma=0.001)

        assert y.shape == (batch_size, n_samples)

    def test_batch_3d(self) -> None:
        """Test Leaky LMS with 3D batched signals."""
        torch.manual_seed(42)

        shape = (2, 3, 100)
        num_taps = 4

        x = torch.randn(*shape, dtype=torch.float64)
        d = torch.randn(*shape, dtype=torch.float64)

        y = leaky_lms(x, d, num_taps=num_taps, mu=0.05, gamma=0.001)

        assert y.shape == shape

    def test_batch_matches_individual(self) -> None:
        """Test that batched result matches processing individually."""
        torch.manual_seed(42)

        batch_size = 3
        n_samples = 50
        num_taps = 3
        mu = 0.05
        gamma = 0.001

        x = torch.randn(batch_size, n_samples, dtype=torch.float64)
        d = torch.randn(batch_size, n_samples, dtype=torch.float64)

        # Batched processing
        y_batch = leaky_lms(x, d, num_taps=num_taps, mu=mu, gamma=gamma)

        # Individual processing
        y_individual = []
        for i in range(batch_size):
            y_i = leaky_lms(x[i], d[i], num_taps=num_taps, mu=mu, gamma=gamma)
            y_individual.append(y_i)
        y_individual = torch.stack(y_individual)

        torch.testing.assert_close(
            y_batch, y_individual, rtol=1e-10, atol=1e-10
        )


class TestLeakyLMSInitialWeights:
    """Test Leaky LMS with initial weight specification."""

    def test_initial_weights_zeros(self) -> None:
        """Test that default initial weights are zeros."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 4

        # Default (zeros)
        y1 = leaky_lms(x, d, num_taps=num_taps, mu=0.05, gamma=0.001)

        # Explicit zeros
        w0 = torch.zeros(num_taps, dtype=torch.float64)
        y2 = leaky_lms(x, d, num_taps=num_taps, mu=0.05, gamma=0.001, w0=w0)

        torch.testing.assert_close(y1, y2, rtol=1e-10, atol=1e-10)

    def test_initial_weights_nonzero(self) -> None:
        """Test Leaky LMS with non-zero initial weights."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 3

        # Non-zero initial weights
        w0 = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)

        y, w = leaky_lms(
            x,
            d,
            num_taps=num_taps,
            mu=0.05,
            gamma=0.001,
            w0=w0,
            return_weights=True,
        )

        assert y.shape == x.shape
        assert w.shape == w0.shape


class TestLeakyLMSReturnWeights:
    """Test Leaky LMS return_weights option."""

    def test_return_weights_false(self) -> None:
        """Test that return_weights=False returns only y."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)

        result = leaky_lms(
            x, d, num_taps=4, mu=0.05, gamma=0.001, return_weights=False
        )

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_return_weights_true(self) -> None:
        """Test that return_weights=True returns (y, w) tuple."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 4

        result = leaky_lms(
            x, d, num_taps=num_taps, mu=0.05, gamma=0.001, return_weights=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

        y, w = result
        assert y.shape == x.shape
        assert w.shape == (num_taps,)


class TestLeakyLMSDtypeAndDevice:
    """Test Leaky LMS dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float32)
        d = torch.randn(100, dtype=torch.float32)

        y = leaky_lms(x, d, num_taps=4, mu=0.05, gamma=0.001)

        assert y.dtype == torch.float32

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)

        y = leaky_lms(x, d, num_taps=4, mu=0.05, gamma=0.001)

        assert y.dtype == torch.float64

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64, device="cpu")
        d = torch.randn(100, dtype=torch.float64, device="cpu")

        y = leaky_lms(x, d, num_taps=4, mu=0.05, gamma=0.001)

        assert y.device == x.device


class TestLeakyLMSEdgeCases:
    """Test Leaky LMS edge cases."""

    def test_single_tap(self) -> None:
        """Test Leaky LMS with single tap (adaptive gain)."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = 0.7 * x  # Desired is just a scaled version

        y, w = leaky_lms(
            x, d, num_taps=1, mu=0.1, gamma=0.001, return_weights=True
        )

        # Weight should converge close to 0.7 (slight bias from leakage)
        assert abs(w[0].item() - 0.7) < 0.15

    def test_short_signal(self) -> None:
        """Test Leaky LMS with short signal."""
        torch.manual_seed(42)

        x = torch.randn(10, dtype=torch.float64)
        d = torch.randn(10, dtype=torch.float64)

        y = leaky_lms(x, d, num_taps=3, mu=0.05, gamma=0.001)

        assert y.shape == x.shape

    def test_num_taps_equals_signal_length(self) -> None:
        """Test when num_taps equals signal length."""
        torch.manual_seed(42)

        n_samples = 10
        x = torch.randn(n_samples, dtype=torch.float64)
        d = torch.randn(n_samples, dtype=torch.float64)

        y = leaky_lms(x, d, num_taps=n_samples, mu=0.01, gamma=0.001)

        assert y.shape == x.shape

    def test_zero_step_size(self) -> None:
        """Test that mu=0 means no weight updates (only leakage)."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 3

        w0 = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)

        # With mu=0, weights should only decay due to leakage
        y, w = leaky_lms(
            x,
            d,
            num_taps=num_taps,
            mu=0.0,
            gamma=0.01,  # Some leakage
            w0=w0,
            return_weights=True,
        )

        # Weights should be decayed but not zero
        # After 100 samples with decay factor (1 - 0.0 * 0.01)^100 = 1.0
        # Actually with mu=0, leakage factor is (1 - mu*gamma) = 1.0
        # So weights remain unchanged
        torch.testing.assert_close(w, w0, rtol=1e-10, atol=1e-10)
