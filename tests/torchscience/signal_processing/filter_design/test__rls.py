"""Tests for RLS (Recursive Least Squares) adaptive filter."""

import torch

from torchscience.signal_processing.filter_design import rls


class TestRLSConverges:
    """Test that RLS converges on system identification problems."""

    def test_rls_converges_simple_system(self) -> None:
        """Test RLS converges to identify a simple FIR system."""
        torch.manual_seed(42)

        # True system (FIR filter)
        h_true = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate input signal
        n_samples = 500
        x = torch.randn(n_samples, dtype=torch.float64)

        # Generate desired signal (output of true system)
        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run RLS
        y, w = rls(x, d, num_taps=num_taps, lam=0.99, return_weights=True)

        # Final weights should be close to true system
        torch.testing.assert_close(w, h_true, rtol=0.1, atol=0.05)

    def test_rls_converges_with_high_forgetting_factor(self) -> None:
        """Test RLS converges with high forgetting factor (slower adaptation)."""
        torch.manual_seed(123)

        # True system
        h_true = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate input signal
        n_samples = 1000
        x = torch.randn(n_samples, dtype=torch.float64)

        # Generate desired signal
        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run RLS with high forgetting factor
        y, w = rls(x, d, num_taps=num_taps, lam=0.999, return_weights=True)

        # Final weights should be close to true system
        torch.testing.assert_close(w, h_true, rtol=0.05, atol=0.02)

    def test_rls_output_reduces_error(self) -> None:
        """Test that RLS output error decreases over time."""
        torch.manual_seed(999)

        # True system
        h_true = torch.tensor([0.8, -0.4], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate signals
        n_samples = 200
        x = torch.randn(n_samples, dtype=torch.float64)

        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run RLS
        y = rls(x, d, num_taps=num_taps, lam=0.99)

        # Compute error in first and second half
        e = d - y
        e_first_half = e[: n_samples // 2].pow(2).mean()
        e_second_half = e[n_samples // 2 :].pow(2).mean()

        # Second half should have smaller error (converged more)
        assert e_second_half < e_first_half


class TestRLSFasterThanLMS:
    """Test that RLS converges faster than LMS."""

    def test_rls_faster_convergence_than_lms(self) -> None:
        """Test that RLS converges faster than LMS on system identification."""
        from torchscience.signal_processing.filter_design import lms

        torch.manual_seed(42)

        # True system - use more taps to make convergence difference visible
        h_true = torch.tensor([0.6, 0.3, 0.1, -0.2, 0.15], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate signals - short signal to see convergence speed difference
        n_samples = 100
        x = torch.randn(n_samples, dtype=torch.float64)

        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run both algorithms with conservative settings for LMS
        y_rls = rls(x, d, num_taps=num_taps, lam=0.99)
        # Use smaller mu for stable LMS - this makes convergence slower
        y_lms = lms(x, d, num_taps=num_taps, mu=0.02)

        # Compute MSE in early convergence phase (first 20-40 samples)
        # This is where RLS advantage is most visible
        early_region = slice(10, 40)
        mse_rls = (d[early_region] - y_rls[early_region]).pow(2).mean()
        mse_lms = (d[early_region] - y_lms[early_region]).pow(2).mean()

        # RLS should have lower error during early convergence
        assert mse_rls < mse_lms


class TestRLSGradients:
    """Test RLS gradient computation."""

    def test_gradcheck_weights_gradients(self) -> None:
        """Gradient check for initial weights."""
        torch.manual_seed(42)

        x = torch.randn(30, dtype=torch.float64)
        d = torch.randn(30, dtype=torch.float64)
        w0 = torch.randn(3, dtype=torch.float64, requires_grad=True)

        def fn(w0_):
            return rls(x, d, num_taps=3, lam=0.99, w0=w0_)

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

        x = torch.randn(20, dtype=torch.float64, requires_grad=True)
        d = torch.randn(20, dtype=torch.float64)

        def fn(x_):
            return rls(x_, d, num_taps=3, lam=0.99)

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

        x = torch.randn(20, dtype=torch.float64)
        d = torch.randn(20, dtype=torch.float64, requires_grad=True)

        def fn(d_):
            return rls(x, d_, num_taps=3, lam=0.99)

        torch.autograd.gradcheck(
            fn,
            (d,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestRLSBatchSignals:
    """Test RLS with batched inputs."""

    def test_batch_2d(self) -> None:
        """Test RLS with 2D batched signals."""
        torch.manual_seed(42)

        batch_size = 4
        n_samples = 50
        num_taps = 3

        x = torch.randn(batch_size, n_samples, dtype=torch.float64)
        d = torch.randn(batch_size, n_samples, dtype=torch.float64)

        y = rls(x, d, num_taps=num_taps, lam=0.99)

        assert y.shape == (batch_size, n_samples)

    def test_batch_3d(self) -> None:
        """Test RLS with 3D batched signals."""
        torch.manual_seed(42)

        shape = (2, 3, 50)
        num_taps = 4

        x = torch.randn(*shape, dtype=torch.float64)
        d = torch.randn(*shape, dtype=torch.float64)

        y = rls(x, d, num_taps=num_taps, lam=0.99)

        assert y.shape == shape

    def test_batch_matches_individual(self) -> None:
        """Test that batched result matches processing individually."""
        torch.manual_seed(42)

        batch_size = 3
        n_samples = 30
        num_taps = 3
        lam = 0.99

        x = torch.randn(batch_size, n_samples, dtype=torch.float64)
        d = torch.randn(batch_size, n_samples, dtype=torch.float64)

        # Batched processing
        y_batch = rls(x, d, num_taps=num_taps, lam=lam)

        # Individual processing
        y_individual = []
        for i in range(batch_size):
            y_i = rls(x[i], d[i], num_taps=num_taps, lam=lam)
            y_individual.append(y_i)
        y_individual = torch.stack(y_individual)

        torch.testing.assert_close(
            y_batch, y_individual, rtol=1e-10, atol=1e-10
        )


class TestRLSInitialWeights:
    """Test RLS with initial weight specification."""

    def test_initial_weights_zeros(self) -> None:
        """Test that default initial weights are zeros."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64)
        d = torch.randn(50, dtype=torch.float64)
        num_taps = 4

        # Default (zeros)
        y1 = rls(x, d, num_taps=num_taps, lam=0.99)

        # Explicit zeros
        w0 = torch.zeros(num_taps, dtype=torch.float64)
        y2 = rls(x, d, num_taps=num_taps, lam=0.99, w0=w0)

        torch.testing.assert_close(y1, y2, rtol=1e-10, atol=1e-10)

    def test_initial_weights_nonzero(self) -> None:
        """Test RLS with non-zero initial weights."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64)
        d = torch.randn(50, dtype=torch.float64)
        num_taps = 3

        # Non-zero initial weights
        w0 = torch.tensor([0.5, 0.3, 0.1], dtype=torch.float64)

        y, w = rls(
            x, d, num_taps=num_taps, lam=0.99, w0=w0, return_weights=True
        )

        assert y.shape == x.shape
        assert w.shape == w0.shape


class TestRLSReturnWeights:
    """Test RLS return_weights option."""

    def test_return_weights_false(self) -> None:
        """Test that return_weights=False returns only y."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64)
        d = torch.randn(50, dtype=torch.float64)

        result = rls(x, d, num_taps=4, lam=0.99, return_weights=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == x.shape

    def test_return_weights_true(self) -> None:
        """Test that return_weights=True returns (y, w) tuple."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64)
        d = torch.randn(50, dtype=torch.float64)
        num_taps = 4

        result = rls(x, d, num_taps=num_taps, lam=0.99, return_weights=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        y, w = result
        assert y.shape == x.shape
        assert w.shape == (num_taps,)


class TestRLSDtypeAndDevice:
    """Test RLS dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float32)
        d = torch.randn(50, dtype=torch.float32)

        y = rls(x, d, num_taps=4, lam=0.99)

        assert y.dtype == torch.float32

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64)
        d = torch.randn(50, dtype=torch.float64)

        y = rls(x, d, num_taps=4, lam=0.99)

        assert y.dtype == torch.float64

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        torch.manual_seed(42)

        x = torch.randn(50, dtype=torch.float64, device="cpu")
        d = torch.randn(50, dtype=torch.float64, device="cpu")

        y = rls(x, d, num_taps=4, lam=0.99)

        assert y.device == x.device


class TestRLSEdgeCases:
    """Test RLS edge cases."""

    def test_single_tap(self) -> None:
        """Test RLS with single tap (adaptive gain)."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = 0.7 * x  # Desired is just a scaled version

        y, w = rls(x, d, num_taps=1, lam=0.99, return_weights=True)

        # Weight should converge to ~0.7
        assert abs(w[0].item() - 0.7) < 0.1

    def test_short_signal(self) -> None:
        """Test RLS with short signal."""
        torch.manual_seed(42)

        x = torch.randn(10, dtype=torch.float64)
        d = torch.randn(10, dtype=torch.float64)

        y = rls(x, d, num_taps=3, lam=0.99)

        assert y.shape == x.shape

    def test_num_taps_equals_signal_length(self) -> None:
        """Test when num_taps equals signal length."""
        torch.manual_seed(42)

        n_samples = 10
        x = torch.randn(n_samples, dtype=torch.float64)
        d = torch.randn(n_samples, dtype=torch.float64)

        y = rls(x, d, num_taps=n_samples, lam=0.99)

        assert y.shape == x.shape


class TestRLSParameters:
    """Test RLS-specific parameters."""

    def test_different_lambda_values(self) -> None:
        """Test RLS with different forgetting factor values."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 4

        # Different lambda values should give different results
        y1 = rls(x, d, num_taps=num_taps, lam=0.95)
        y2 = rls(x, d, num_taps=num_taps, lam=0.99)
        y3 = rls(x, d, num_taps=num_taps, lam=1.0)

        # Results should differ
        assert not torch.allclose(y1, y2)
        assert not torch.allclose(y2, y3)

    def test_different_delta_values(self) -> None:
        """Test RLS with different delta (initial P scaling) values."""
        torch.manual_seed(42)

        x = torch.randn(100, dtype=torch.float64)
        d = torch.randn(100, dtype=torch.float64)
        num_taps = 4

        # Different delta values should give different initial behavior
        y1 = rls(x, d, num_taps=num_taps, lam=0.99, delta=0.1)
        y2 = rls(x, d, num_taps=num_taps, lam=0.99, delta=1.0)
        y3 = rls(x, d, num_taps=num_taps, lam=0.99, delta=100.0)

        # First few samples should differ due to different P initialization
        assert not torch.allclose(y1[:10], y2[:10])
        assert not torch.allclose(y2[:10], y3[:10])

    def test_lambda_one_no_forgetting(self) -> None:
        """Test that lambda=1.0 means no forgetting (standard LS)."""
        torch.manual_seed(42)

        # True system
        h_true = torch.tensor([0.5, 0.3], dtype=torch.float64)
        num_taps = len(h_true)

        # Generate signals
        n_samples = 200
        x = torch.randn(n_samples, dtype=torch.float64)

        d = torch.zeros(n_samples, dtype=torch.float64)
        for n in range(n_samples):
            for k in range(num_taps):
                if n - k >= 0:
                    d[n] += h_true[k] * x[n - k]

        # Run RLS with lambda=1.0 (no forgetting)
        y, w = rls(x, d, num_taps=num_taps, lam=1.0, return_weights=True)

        # Should still converge to true weights
        torch.testing.assert_close(w, h_true, rtol=0.1, atol=0.05)
