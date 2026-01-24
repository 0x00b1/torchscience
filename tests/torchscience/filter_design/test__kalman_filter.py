"""Tests for Kalman filter for state estimation."""

import torch

from torchscience.filter_design import kalman_filter


class TestKalmanFilterConstantVelocity:
    """Test Kalman filter for constant velocity tracking."""

    def test_constant_velocity_position_tracking(self) -> None:
        """Test tracking a constant velocity target in 1D."""
        torch.manual_seed(42)

        # State: [position, velocity]
        # True trajectory: starts at 0, moves with velocity 1.0
        n_samples = 100
        dt = 1.0
        true_velocity = 1.0
        true_positions = (
            torch.arange(n_samples, dtype=torch.float64) * dt * true_velocity
        )

        # Add measurement noise
        noise_std = 0.5
        z = true_positions + noise_std * torch.randn(
            n_samples, dtype=torch.float64
        )
        z = z.unsqueeze(-1)  # Shape: (n_samples, 1)

        # State transition matrix (constant velocity model)
        # x_{k+1} = [1, dt; 0, 1] @ x_k
        F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=torch.float64)

        # Observation matrix (we only observe position)
        H = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

        # Process noise covariance (small - we trust our model)
        Q = 0.01 * torch.eye(2, dtype=torch.float64)

        # Observation noise covariance
        R = torch.tensor([[noise_std**2]], dtype=torch.float64)

        # Initial state and covariance
        x0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        P0 = torch.eye(2, dtype=torch.float64)

        # Run Kalman filter
        x_hat, P_final = kalman_filter(
            z, H, R, F=F, Q=Q, x0=x0, P0=P0, return_covariance=True
        )

        # Check output shapes
        assert x_hat.shape == (n_samples, 2)
        assert P_final.shape == (2, 2)

        # Check that estimated positions track true positions well
        estimated_positions = x_hat[:, 0]
        final_position_error = torch.abs(
            estimated_positions[-1] - true_positions[-1]
        )
        assert final_position_error < 2.0  # Should be close after convergence

        # Check that estimated velocity converges to true velocity
        estimated_velocity = x_hat[-1, 1]
        velocity_error = torch.abs(estimated_velocity - true_velocity)
        assert velocity_error < 0.2  # Should converge to ~1.0

    def test_constant_velocity_2d(self) -> None:
        """Test tracking a constant velocity target in 2D."""
        torch.manual_seed(123)

        n_samples = 50
        dt = 0.1

        # True trajectory in 2D
        true_vx, true_vy = 2.0, -1.0
        t = torch.arange(n_samples, dtype=torch.float64) * dt
        true_x = 10.0 + true_vx * t
        true_y = 5.0 + true_vy * t

        # Measurements with noise
        noise_std = 0.3
        z_x = true_x + noise_std * torch.randn(n_samples, dtype=torch.float64)
        z_y = true_y + noise_std * torch.randn(n_samples, dtype=torch.float64)
        z = torch.stack([z_x, z_y], dim=-1)  # Shape: (n_samples, 2)

        # State: [x, y, vx, vy]
        n_state = 4

        # State transition
        F = torch.tensor(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        # Observation matrix (observe x and y positions)
        H = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float64
        )

        # Noise covariances
        Q = 0.001 * torch.eye(n_state, dtype=torch.float64)
        R = noise_std**2 * torch.eye(2, dtype=torch.float64)

        # Initial state
        x0 = torch.tensor([10.0, 5.0, 0.0, 0.0], dtype=torch.float64)
        P0 = torch.eye(n_state, dtype=torch.float64)

        # Run filter
        x_hat = kalman_filter(z, H, R, F=F, Q=Q, x0=x0, P0=P0)

        assert x_hat.shape == (n_samples, n_state)

        # Check velocity convergence
        final_vx = x_hat[-1, 2]
        final_vy = x_hat[-1, 3]
        assert torch.abs(final_vx - true_vx) < 0.5
        assert torch.abs(final_vy - true_vy) < 0.5


class TestKalmanFilterNoiseReduction:
    """Test Kalman filter for noise reduction / smoothing."""

    def test_noise_reduction_static_target(self) -> None:
        """Test noise reduction when tracking a static target."""
        torch.manual_seed(456)

        n_samples = 100
        true_value = 5.0
        noise_std = 2.0

        # Noisy measurements of a constant value
        z = true_value + noise_std * torch.randn(
            n_samples, dtype=torch.float64
        )
        z = z.unsqueeze(-1)  # Shape: (n_samples, 1)

        # State is just the value itself
        H = torch.tensor([[1.0]], dtype=torch.float64)
        R = torch.tensor([[noise_std**2]], dtype=torch.float64)

        # No state transition (or identity)
        F = torch.tensor([[1.0]], dtype=torch.float64)
        Q = torch.tensor(
            [[0.001]], dtype=torch.float64
        )  # Very small process noise

        x0 = torch.tensor([0.0], dtype=torch.float64)
        P0 = torch.tensor(
            [[10.0]], dtype=torch.float64
        )  # Large initial uncertainty

        x_hat = kalman_filter(z, H, R, F=F, Q=Q, x0=x0, P0=P0)

        # Output should have reduced variance compared to input
        input_var = z.var()
        output_var = x_hat.var()
        assert output_var < input_var

        # Final estimate should be close to true value
        final_estimate = x_hat[-1, 0]
        assert torch.abs(final_estimate - true_value) < 0.5

    def test_noise_reduction_improves_snr(self) -> None:
        """Test that Kalman filter improves signal-to-noise ratio."""
        torch.manual_seed(789)

        n_samples = 200
        # True signal: slow sinusoid
        t = torch.linspace(0, 4 * torch.pi, n_samples, dtype=torch.float64)
        true_signal = torch.sin(t)
        noise_std = 0.5

        # Noisy measurements
        z = true_signal + noise_std * torch.randn(
            n_samples, dtype=torch.float64
        )
        z = z.unsqueeze(-1)

        # Simple state = value model
        H = torch.tensor([[1.0]], dtype=torch.float64)
        R = torch.tensor([[noise_std**2]], dtype=torch.float64)
        F = torch.tensor([[1.0]], dtype=torch.float64)
        Q = torch.tensor([[0.01]], dtype=torch.float64)

        x_hat = kalman_filter(z, H, R, F=F, Q=Q)

        # Compute MSE before and after filtering
        mse_before = ((z.squeeze(-1) - true_signal) ** 2).mean()
        mse_after = ((x_hat.squeeze(-1) - true_signal) ** 2).mean()

        # Filtering should reduce MSE
        assert mse_after < mse_before


class TestKalmanFilterGradcheck:
    """Test gradient computation through Kalman filter."""

    def test_gradcheck_observations(self) -> None:
        """Gradient check for observations."""
        torch.manual_seed(42)

        n_samples = 10
        n_obs = 2
        n_state = 2

        z = torch.randn(
            n_samples, n_obs, dtype=torch.float64, requires_grad=True
        )
        H = torch.randn(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)
        F = torch.eye(n_state, dtype=torch.float64)
        Q = 0.1 * torch.eye(n_state, dtype=torch.float64)

        def fn(z_):
            return kalman_filter(z_, H, R, F=F, Q=Q)

        torch.autograd.gradcheck(
            fn,
            (z,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_initial_state(self) -> None:
        """Gradient check for initial state."""
        torch.manual_seed(42)

        n_samples = 10
        n_obs = 2
        n_state = 3

        z = torch.randn(n_samples, n_obs, dtype=torch.float64)
        H = torch.randn(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)
        F = torch.eye(n_state, dtype=torch.float64)
        Q = 0.1 * torch.eye(n_state, dtype=torch.float64)
        x0 = torch.randn(n_state, dtype=torch.float64, requires_grad=True)

        def fn(x0_):
            return kalman_filter(z, H, R, F=F, Q=Q, x0=x0_)

        torch.autograd.gradcheck(
            fn,
            (x0,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_gradcheck_h_matrix(self) -> None:
        """Gradient check for observation matrix H."""
        torch.manual_seed(42)

        n_samples = 8
        n_obs = 2
        n_state = 2

        z = torch.randn(n_samples, n_obs, dtype=torch.float64)
        H = torch.randn(
            n_obs, n_state, dtype=torch.float64, requires_grad=True
        )
        R = torch.eye(n_obs, dtype=torch.float64)
        F = torch.eye(n_state, dtype=torch.float64)
        Q = 0.1 * torch.eye(n_state, dtype=torch.float64)

        def fn(H_):
            return kalman_filter(z, H_, R, F=F, Q=Q)

        torch.autograd.gradcheck(
            fn,
            (H,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-4,
        )


class TestKalmanFilterBatchInputs:
    """Test Kalman filter with batched inputs."""

    def test_batch_2d(self) -> None:
        """Test with 2D batch of observations."""
        torch.manual_seed(42)

        batch_size = 4
        n_samples = 20
        n_obs = 2
        n_state = 3

        z = torch.randn(batch_size, n_samples, n_obs, dtype=torch.float64)
        H = torch.randn(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)

        x_hat = kalman_filter(z, H, R)

        assert x_hat.shape == (batch_size, n_samples, n_state)

    def test_batch_3d(self) -> None:
        """Test with 3D batch of observations."""
        torch.manual_seed(42)

        shape = (2, 3, 15, 2)  # (batch1, batch2, n_samples, n_obs)
        n_state = 2

        z = torch.randn(*shape, dtype=torch.float64)
        H = torch.eye(2, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)

        x_hat = kalman_filter(z, H, R)

        assert x_hat.shape == (2, 3, 15, n_state)

    def test_batch_matches_individual(self) -> None:
        """Test that batched result matches processing individually."""
        torch.manual_seed(42)

        batch_size = 3
        n_samples = 15
        n_obs = 2
        n_state = 2

        z = torch.randn(batch_size, n_samples, n_obs, dtype=torch.float64)
        H = torch.randn(n_obs, n_state, dtype=torch.float64)
        R = 0.1 * torch.eye(n_obs, dtype=torch.float64)
        F = torch.eye(n_state, dtype=torch.float64)
        Q = 0.01 * torch.eye(n_state, dtype=torch.float64)

        # Batched processing
        x_batch = kalman_filter(z, H, R, F=F, Q=Q)

        # Individual processing
        x_individual = []
        for i in range(batch_size):
            x_i = kalman_filter(z[i], H, R, F=F, Q=Q)
            x_individual.append(x_i)
        x_individual = torch.stack(x_individual)

        torch.testing.assert_close(
            x_batch, x_individual, rtol=1e-10, atol=1e-10
        )


class TestKalmanFilterDefaults:
    """Test Kalman filter default parameter handling."""

    def test_default_f_is_identity(self) -> None:
        """Test that default F is identity matrix."""
        torch.manual_seed(42)

        n_samples = 20
        n_obs = 2
        n_state = 2

        z = torch.randn(n_samples, n_obs, dtype=torch.float64)
        H = torch.eye(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)

        # Without F (should default to identity)
        x1 = kalman_filter(z, H, R)

        # With explicit identity F
        F = torch.eye(n_state, dtype=torch.float64)
        x2 = kalman_filter(z, H, R, F=F)

        torch.testing.assert_close(x1, x2, rtol=1e-10, atol=1e-10)

    def test_default_q_is_zeros(self) -> None:
        """Test that default Q is zeros matrix."""
        torch.manual_seed(42)

        n_samples = 20
        n_obs = 2
        n_state = 2

        z = torch.randn(n_samples, n_obs, dtype=torch.float64)
        H = torch.eye(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)
        F = torch.eye(n_state, dtype=torch.float64)

        # Without Q (should default to zeros)
        x1 = kalman_filter(z, H, R, F=F)

        # With explicit zeros Q
        Q = torch.zeros(n_state, n_state, dtype=torch.float64)
        x2 = kalman_filter(z, H, R, F=F, Q=Q)

        torch.testing.assert_close(x1, x2, rtol=1e-10, atol=1e-10)

    def test_default_x0_is_zeros(self) -> None:
        """Test that default x0 is zeros."""
        torch.manual_seed(42)

        n_samples = 20
        n_obs = 2
        n_state = 2

        z = torch.randn(n_samples, n_obs, dtype=torch.float64)
        H = torch.eye(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)

        # Without x0 (should default to zeros)
        x1 = kalman_filter(z, H, R)

        # With explicit zeros x0
        x0 = torch.zeros(n_state, dtype=torch.float64)
        x2 = kalman_filter(z, H, R, x0=x0)

        torch.testing.assert_close(x1, x2, rtol=1e-10, atol=1e-10)

    def test_default_p0_is_identity(self) -> None:
        """Test that default P0 is identity matrix."""
        torch.manual_seed(42)

        n_samples = 20
        n_obs = 2
        n_state = 2

        z = torch.randn(n_samples, n_obs, dtype=torch.float64)
        H = torch.eye(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)

        # Without P0 (should default to identity)
        x1 = kalman_filter(z, H, R)

        # With explicit identity P0
        P0 = torch.eye(n_state, dtype=torch.float64)
        x2 = kalman_filter(z, H, R, P0=P0)

        torch.testing.assert_close(x1, x2, rtol=1e-10, atol=1e-10)


class TestKalmanFilterReturnCovariance:
    """Test return_covariance option."""

    def test_return_covariance_false(self) -> None:
        """Test that return_covariance=False returns only x_hat."""
        torch.manual_seed(42)

        z = torch.randn(20, 2, dtype=torch.float64)
        H = torch.eye(2, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)

        result = kalman_filter(z, H, R, return_covariance=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (20, 2)

    def test_return_covariance_true(self) -> None:
        """Test that return_covariance=True returns (x_hat, P)."""
        torch.manual_seed(42)

        n_samples = 20
        n_state = 3

        z = torch.randn(n_samples, 2, dtype=torch.float64)
        H = torch.randn(2, n_state, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)

        result = kalman_filter(z, H, R, return_covariance=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        x_hat, P = result
        assert x_hat.shape == (n_samples, n_state)
        assert P.shape == (n_state, n_state)

    def test_covariance_is_positive_definite(self) -> None:
        """Test that returned covariance is positive definite."""
        torch.manual_seed(42)

        z = torch.randn(50, 2, dtype=torch.float64)
        H = torch.eye(2, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)
        Q = 0.1 * torch.eye(2, dtype=torch.float64)

        _, P = kalman_filter(z, H, R, Q=Q, return_covariance=True)

        # Positive definite: all eigenvalues > 0
        eigenvalues = torch.linalg.eigvalsh(P)
        assert (eigenvalues > 0).all()


class TestKalmanFilterDtypeDevice:
    """Test dtype and device handling."""

    def test_dtype_float32(self) -> None:
        """Test with float32 inputs."""
        torch.manual_seed(42)

        z = torch.randn(20, 2, dtype=torch.float32)
        H = torch.eye(2, dtype=torch.float32)
        R = torch.eye(2, dtype=torch.float32)

        x_hat = kalman_filter(z, H, R)

        assert x_hat.dtype == torch.float32

    def test_dtype_float64(self) -> None:
        """Test with float64 inputs."""
        torch.manual_seed(42)

        z = torch.randn(20, 2, dtype=torch.float64)
        H = torch.eye(2, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)

        x_hat = kalman_filter(z, H, R)

        assert x_hat.dtype == torch.float64

    def test_device_cpu(self) -> None:
        """Test device preservation (CPU)."""
        torch.manual_seed(42)

        z = torch.randn(20, 2, dtype=torch.float64, device="cpu")
        H = torch.eye(2, dtype=torch.float64, device="cpu")
        R = torch.eye(2, dtype=torch.float64, device="cpu")

        x_hat = kalman_filter(z, H, R)

        assert x_hat.device == z.device


class TestKalmanFilterEdgeCases:
    """Test edge cases."""

    def test_single_sample(self) -> None:
        """Test with single observation."""
        torch.manual_seed(42)

        z = torch.randn(1, 2, dtype=torch.float64)
        H = torch.eye(2, dtype=torch.float64)
        R = torch.eye(2, dtype=torch.float64)

        x_hat = kalman_filter(z, H, R)

        assert x_hat.shape == (1, 2)

    def test_single_state_dimension(self) -> None:
        """Test with single state dimension."""
        torch.manual_seed(42)

        z = torch.randn(20, 1, dtype=torch.float64)
        H = torch.tensor([[1.0]], dtype=torch.float64)
        R = torch.tensor([[0.1]], dtype=torch.float64)

        x_hat = kalman_filter(z, H, R)

        assert x_hat.shape == (20, 1)

    def test_more_states_than_observations(self) -> None:
        """Test when state dimension > observation dimension."""
        torch.manual_seed(42)

        n_samples = 30
        n_obs = 2
        n_state = 4

        z = torch.randn(n_samples, n_obs, dtype=torch.float64)
        H = torch.randn(n_obs, n_state, dtype=torch.float64)
        R = torch.eye(n_obs, dtype=torch.float64)

        x_hat = kalman_filter(z, H, R)

        assert x_hat.shape == (n_samples, n_state)
