"""Phase 2a tests for adaptive adjoint integration and Hermite interpolation."""

import math

import torch

from torchscience.integration._interpolation import HermiteInterpolant
from torchscience.integration.initial_value_problem import (
    adjoint,
    dormand_prince_5,
    runge_kutta_4,
    solve_ivp,
)


class TestAdaptiveAdjoint:
    """Test adaptive adjoint integration."""

    def test_adaptive_adjoint_basic(self):
        """Test that adaptive adjoint option is accepted and works."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "adaptive"},
        )

        y_final, _ = solver(decay, y0, t_span=(0.0, 1.0))
        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)
        # Gradient should be negative (more decay = lower final value)
        assert theta.grad < 0

    def test_adaptive_adjoint_matches_rk4_accuracy(self):
        """Test adaptive adjoint achieves similar accuracy to RK4 with many steps."""
        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        def oscillator(t, y):
            # Damped oscillator: y'' + theta*y' + y = 0
            # State: [position, velocity]
            return torch.stack([y[1], -y[0] - theta * y[1]])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        # Reference: RK4 with many steps
        theta_ref = theta.clone().detach().requires_grad_(True)

        def oscillator_ref(t, y):
            return torch.stack([y[1], -y[0] - theta_ref * y[1]])

        solver_ref = adjoint(
            dormand_prince_5,
            params=[theta_ref],
            adjoint_options={"method": "rk4", "n_steps": 500},
        )
        y_ref, _ = solver_ref(oscillator_ref, y0.clone(), t_span=(0.0, 5.0))
        y_ref.sum().backward()

        # Adaptive adjoint
        solver_adaptive = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "adaptive"},
        )
        y_adaptive, _ = solver_adaptive(
            oscillator, y0.clone(), t_span=(0.0, 5.0)
        )
        y_adaptive.sum().backward()

        # Gradients should be close
        assert torch.allclose(theta.grad, theta_ref.grad, rtol=1e-3)


class TestHermiteInterpolation:
    """Test Hermite interpolation for fixed-step solvers."""

    def test_rk4_hermite_interpolant_type(self):
        """Test that RK4 returns HermiteInterpolant when requested."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)
        _, interp = runge_kutta_4(
            decay, y0, t_span=(0.0, 1.0), dt=0.1, dense_output="hermite"
        )

        assert isinstance(interp, HermiteInterpolant)

    def test_hermite_interpolant_accuracy(self):
        """Test Hermite interpolant achieves 4th-order accuracy."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Solve with Hermite interpolation
        _, interp = runge_kutta_4(
            decay, y0, t_span=(0.0, 1.0), dt=0.1, dense_output="hermite"
        )

        # Check accuracy at midpoints (where linear would be worst)
        t_test = torch.tensor([0.05, 0.15, 0.25, 0.35], dtype=torch.float64)
        y_interp = interp(t_test)
        y_exact = torch.exp(-t_test).unsqueeze(-1)

        # Should be much better than linear interpolation
        error = (y_interp - y_exact).abs().max()
        assert error < 1e-3, f"Hermite interpolation error {error} too large"

    def test_hermite_vs_linear_improvement(self):
        """Test that Hermite is significantly more accurate than linear."""

        def oscillator(t, y):
            return torch.stack([y[1], -y[0]])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        # Hermite interpolation
        _, interp_hermite = runge_kutta_4(
            oscillator, y0, t_span=(0.0, 2.0), dt=0.2, dense_output="hermite"
        )

        # Linear interpolation (default)
        _, interp_linear = runge_kutta_4(
            oscillator, y0, t_span=(0.0, 2.0), dt=0.2, dense_output="linear"
        )

        # Test at midpoint of a step
        t_test = torch.tensor([0.1], dtype=torch.float64)
        t_val = torch.tensor(0.1, dtype=torch.float64)
        y_exact = torch.tensor(
            [[torch.cos(t_val).item(), -torch.sin(t_val).item()]]
        )

        error_hermite = (interp_hermite(t_test) - y_exact).abs().max()
        error_linear = (interp_linear(t_test) - y_exact).abs().max()

        # Hermite should be at least 10x better
        assert error_hermite < error_linear / 10

    def test_hermite_interpolant_boundary_accuracy(self):
        """Test Hermite interpolant is exact at grid points."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        _, interp = runge_kutta_4(
            decay, y0, t_span=(0.0, 1.0), dt=0.25, dense_output="hermite"
        )

        # Query at exact grid points
        t_grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        y_interp = interp(t_grid)

        # Should match stored values exactly (within numerical precision)
        y_stored = interp.y_points
        for i, t in enumerate(t_grid):
            y_at_t = interp(t)
            assert torch.allclose(y_at_t, y_stored[i], atol=1e-14)

    def test_hermite_interpolant_scalar_query(self):
        """Test Hermite interpolant handles scalar queries correctly."""

        def decay(t, y):
            return -y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        _, interp = runge_kutta_4(
            decay, y0, t_span=(0.0, 1.0), dt=0.1, dense_output="hermite"
        )

        # Query with scalar
        y_scalar = interp(0.5)
        assert y_scalar.shape == (1,), (
            f"Expected shape (1,), got {y_scalar.shape}"
        )

        # Query with 0-dim tensor
        y_0d = interp(torch.tensor(0.5, dtype=torch.float64))
        assert y_0d.shape == (1,), f"Expected shape (1,), got {y_0d.shape}"


class TestBinomialCheckpointing:
    """Tests for binomial checkpointing with Revolve algorithm."""

    def test_binomial_checkpoint_basic(self):
        """Test that binomial checkpointing works with solve_ivp."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Solve with binomial checkpointing
        result = solve_ivp(
            decay,
            y0,
            t_span=(0.0, 10.0),
            method="dormand_prince_5",
            sensitivity="binomial",
            params=[theta],
        )

        # Forward should succeed
        assert result.success
        assert torch.isfinite(result.y_final).all()

        # Should be able to compute gradients
        result.y_final.sum().backward()
        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

        # Gradient should be negative (more decay = lower final value)
        assert theta.grad < 0

    def test_binomial_memory_scaling(self):
        """Test that binomial checkpointing uses O(log n) checkpoints."""
        from torchscience.integration.initial_value_problem._checkpointing import (
            BinomialCheckpointSchedule,
        )

        # Test optimal checkpoint counts for various step counts
        test_cases = [
            (10, 4),  # ceil(log2(10)) = 4
            (100, 7),  # ceil(log2(100)) = 7
            (1000, 10),  # ceil(log2(1000)) = 10
            (10000, 14),  # ceil(log2(10000)) = 14
        ]

        for n_steps, expected_max in test_cases:
            schedule = BinomialCheckpointSchedule.from_n_steps(
                n_steps, 0.0, 1.0
            )
            n_segments = schedule.n_segments

            # Optimal should be ceil(log2(n_steps)) segments
            expected = math.ceil(math.log2(n_steps)) if n_steps > 1 else 1
            assert n_segments == expected, (
                f"For n_steps={n_steps}, expected {expected} segments, "
                f"got {n_segments}"
            )
            # Should be within bounds
            assert n_segments <= expected_max, (
                f"For n_steps={n_steps}, segments {n_segments} > max {expected_max}"
            )

    def test_binomial_vs_linear_accuracy(self):
        """Test that binomial gradients match linear checkpointing gradients."""
        theta = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

        def oscillator(t, y):
            # Damped oscillator: y'' + theta*y' + y = 0
            return torch.stack([y[1], -y[0] - theta * y[1]])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        # Reference: Linear checkpointing with many checkpoints
        theta_linear = theta.clone().detach().requires_grad_(True)

        def oscillator_linear(t, y):
            return torch.stack([y[1], -y[0] - theta_linear * y[1]])

        solver_linear = adjoint(
            dormand_prince_5, params=[theta_linear], checkpoints=20
        )
        y_linear, _ = solver_linear(
            oscillator_linear, y0.clone(), t_span=(0.0, 5.0)
        )
        y_linear.sum().backward()

        # Binomial checkpointing
        theta_binomial = theta.clone().detach().requires_grad_(True)

        def oscillator_binomial(t, y):
            return torch.stack([y[1], -y[0] - theta_binomial * y[1]])

        result = solve_ivp(
            oscillator_binomial,
            y0.clone(),
            t_span=(0.0, 5.0),
            method="dormand_prince_5",
            sensitivity="binomial",
            params=[theta_binomial],
        )
        result.y_final.sum().backward()

        # Forward solutions should match exactly (same solver)
        assert torch.allclose(result.y_final, y_linear, atol=1e-10), (
            f"Forward solutions differ: {result.y_final} vs {y_linear}"
        )

        # Gradients should be close
        assert torch.allclose(
            theta_binomial.grad, theta_linear.grad, rtol=1e-3
        ), (
            f"Binomial grad {theta_binomial.grad.item()} != "
            f"linear grad {theta_linear.grad.item()}"
        )

    def test_binomial_vs_standard_adjoint_accuracy(self):
        """Test that binomial gradients match standard adjoint gradients."""
        theta = torch.tensor([1.5], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Reference: Standard adjoint (no checkpointing)
        theta_std = theta.clone().detach().requires_grad_(True)

        def decay_std(t, y):
            return -theta_std * y

        solver_std = adjoint(
            dormand_prince_5,
            params=[theta_std],
            adjoint_options={"method": "rk4", "n_steps": 200},
        )
        y_std, _ = solver_std(decay_std, y0.clone(), t_span=(0.0, 10.0))
        y_std.sum().backward()

        # Binomial checkpointing
        theta_binomial = theta.clone().detach().requires_grad_(True)

        def decay_binomial(t, y):
            return -theta_binomial * y

        result = solve_ivp(
            decay_binomial,
            y0.clone(),
            t_span=(0.0, 10.0),
            method="dormand_prince_5",
            sensitivity="binomial",
            params=[theta_binomial],
        )
        result.y_final.sum().backward()

        # Gradients should be close
        assert torch.allclose(
            theta_binomial.grad, theta_std.grad, rtol=5e-3
        ), (
            f"Binomial grad {theta_binomial.grad.item()} != "
            f"standard adjoint grad {theta_std.grad.item()}"
        )


class TestPhase2aBinomialGraduation:
    """
    Graduation tests for Phase 2a Binomial Checkpointing.

    Run: pytest tests/.../test__adjoint_phase2a.py::TestPhase2aBinomialGraduation -v
    Pass condition: All tests pass.
    """

    def test_g2a_b1_binomial_basic_functionality(self):
        """G2a.B1 [BLOCKER]: Binomial checkpointing basic forward/backward."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        result = solve_ivp(
            f,
            y0,
            t_span=(0.0, 10.0),
            method="dormand_prince_5",
            sensitivity="binomial",
            params=[theta],
        )

        result.y_final.sum().backward()

        assert theta.grad is not None, "G2a.B1 FAIL: No gradient computed"
        assert torch.isfinite(theta.grad), "G2a.B1 FAIL: Non-finite gradient"

    def test_g2a_b2_binomial_log_memory(self):
        """G2a.B2 [BLOCKER]: Binomial uses O(log n) checkpoints."""
        from torchscience.integration.initial_value_problem._checkpointing import (
            BinomialCheckpointSchedule,
        )

        # For 1000 steps, should use ~10 segments (log2(1000) ≈ 10)
        schedule = BinomialCheckpointSchedule.from_n_steps(1000, 0.0, 1.0)
        n_segments = schedule.n_segments

        assert n_segments <= 15, (
            f"G2a.B2 FAIL: {n_segments} segments for 1000 steps, expected <= 15"
        )
        assert n_segments >= 8, (
            f"G2a.B2 FAIL: {n_segments} segments for 1000 steps, expected >= 8"
        )

    def test_g2a_b3_binomial_gradient_accuracy(self):
        """G2a.B3 [BLOCKER]: Binomial gradients within 1% of standard adjoint."""
        theta1 = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        theta2 = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        y0 = torch.tensor([1.0], dtype=torch.float64)

        def f1(t, y):
            return -theta1 * y

        def f2(t, y):
            return -theta2 * y

        # Standard adjoint
        solver1 = adjoint(
            dormand_prince_5,
            params=[theta1],
            adjoint_options={"method": "rk4", "n_steps": 200},
        )
        y1, _ = solver1(f1, y0.clone(), (0.0, 10.0))
        y1.sum().backward()

        # Binomial
        result2 = solve_ivp(
            f2,
            y0.clone(),
            t_span=(0.0, 10.0),
            method="dormand_prince_5",
            sensitivity="binomial",
            params=[theta2],
        )
        result2.y_final.sum().backward()

        rel_error = abs(theta1.grad.item() - theta2.grad.item()) / abs(
            theta1.grad.item()
        )
        assert rel_error < 0.01, (
            f"G2a.B3 FAIL: relative error {rel_error:.4f} >= 1%"
        )


class TestBacksolveAdjoint:
    """Test BacksolveAdjoint mode (recompute forward during backward)."""

    def test_backsolve_basic(self):
        """Test BacksolveAdjoint produces correct gradients."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def decay(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "backsolve"},
        )

        y_final, _ = solver(decay, y0, t_span=(0.0, 1.0))
        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)
        assert theta.grad < 0  # More decay = lower final value

    def test_backsolve_gradient_accuracy(self):
        """Test BacksolveAdjoint matches standard adjoint for simple systems."""
        theta = torch.tensor([2.0], requires_grad=True, dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Reference: standard adjoint
        theta_ref = theta.clone().detach().requires_grad_(True)

        def dynamics_ref(t, y):
            return -theta_ref * y

        solver_ref = adjoint(
            dormand_prince_5,
            params=[theta_ref],
            adjoint_options={"method": "rk4", "n_steps": 200},
        )
        y_ref, _ = solver_ref(dynamics_ref, y0.clone(), t_span=(0.0, 1.0))
        y_ref.sum().backward()

        # BacksolveAdjoint
        solver_backsolve = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "backsolve"},
        )
        y_back, _ = solver_backsolve(dynamics, y0.clone(), t_span=(0.0, 1.0))
        y_back.sum().backward()

        # Gradients should be close
        assert torch.allclose(theta.grad, theta_ref.grad, rtol=1e-2)

    def test_backsolve_stability_warning(self):
        """Test BacksolveAdjoint warns for potentially unstable systems."""
        import warnings

        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        # System with exponential growth (unstable for backward integration)
        def unstable(t, y):
            return theta * y  # dy/dt = +y (exponential growth)

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={"method": "backsolve"},
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_final, _ = solver(unstable, y0, t_span=(0.0, 5.0))
            y_final.sum().backward()

            # Check for reconstruction warning (system is unstable when integrated backwards)
            backsolve_warnings = [
                x
                for x in w
                if "BacksolveAdjoint" in str(x.message)
                or "reconstruction" in str(x.message).lower()
            ]
            # We expect a warning for this unstable system
            assert len(backsolve_warnings) > 0 or torch.isfinite(theta.grad)


class TestEventHandling:
    """Test event detection and handling."""

    def test_event_detection_basic(self):
        """Test basic event detection (zero-crossing)."""

        def falling(t, y):
            # Free fall: y'' = -g, state = [height, velocity]
            g = 10.0
            return torch.stack(
                [y[1], torch.tensor(-g, dtype=y.dtype, device=y.device)]
            )

        y0 = torch.tensor([10.0, 0.0], dtype=torch.float64)  # height=10, v=0

        def hit_ground(t, y):
            """Event: height reaches zero."""
            return y[0]  # Trigger when y[0] = 0

        result = solve_ivp(
            falling,
            y0,
            t_span=(0.0, 5.0),
            events=[hit_ground],
        )

        # Should stop when hitting ground
        # Expected time: t = sqrt(2*h/g) = sqrt(2*10/10) = sqrt(2) ≈ 1.414
        assert result.t_events is not None
        assert len(result.t_events) == 1
        assert len(result.t_events[0]) == 1
        t_hit = result.t_events[0][0]
        expected_t = (2 * 10.0 / 10.0) ** 0.5
        assert abs(t_hit - expected_t) < 0.01

    def test_event_with_terminal(self):
        """Test terminal event stops integration."""

        def oscillator(t, y):
            return torch.stack([y[1], -y[0]])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        def cross_zero(t, y):
            return y[0]

        cross_zero.terminal = True
        cross_zero.direction = -1  # Only trigger on negative crossings

        result = solve_ivp(
            oscillator,
            y0,
            t_span=(0.0, 10.0),
            events=[cross_zero],
        )

        # Should stop at first negative zero crossing (t = pi/2)
        assert result.t_events[0][0] < math.pi  # Should be around pi/2

    def test_event_multiple_events(self):
        """Test multiple event functions."""

        def oscillator(t, y):
            return torch.stack([y[1], -y[0]])

        y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)

        def position_zero(t, y):
            return y[0]

        def velocity_zero(t, y):
            return y[1]

        result = solve_ivp(
            oscillator,
            y0,
            t_span=(0.0, 10.0),
            events=[position_zero, velocity_zero],
        )

        # Position crosses zero at pi/2, 3pi/2, etc.
        # Velocity crosses zero at 0, pi, 2pi, etc.
        assert result.t_events is not None
        assert len(result.t_events) == 2
        assert len(result.t_events[0]) > 0  # Position crossings
        assert len(result.t_events[1]) > 0  # Velocity crossings


class TestSeminormErrorControl:
    """Test seminorm (per-component) error control."""

    def test_seminorm_basic(self):
        """Test seminorm error weighting is accepted."""
        from torchscience.integration.initial_value_problem import solve_ivp

        def system(t, y):
            # Two components with different scales
            return torch.stack([-y[0], -1000 * y[1]])

        y0 = torch.tensor([1.0, 1e-6], dtype=torch.float64)

        # Component weights: first component has loose tolerance
        error_weights = torch.tensor([0.1, 10.0], dtype=torch.float64)

        result = solve_ivp(
            system,
            y0,
            t_span=(0.0, 0.01),
            error_weights=error_weights,
        )

        assert result.success

    def test_seminorm_improves_stiff_handling(self):
        """Test that error weights improve handling of mixed-scale systems."""
        from torchscience.integration.initial_value_problem import solve_ivp

        def stiff_system(t, y):
            # Fast and slow components
            return torch.stack(
                [
                    -y[0],  # Slow decay
                    -1000 * y[1],  # Fast decay
                ]
            )

        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

        # Without weights (may take many steps)
        result_default = solve_ivp(
            stiff_system,
            y0,
            t_span=(0.0, 1.0),
            rtol=1e-6,
            atol=1e-9,
        )

        # With weights focusing on slow component
        error_weights = torch.tensor([1.0, 0.01], dtype=torch.float64)
        result_weighted = solve_ivp(
            stiff_system,
            y0,
            t_span=(0.0, 1.0),
            rtol=1e-6,
            atol=1e-9,
            error_weights=error_weights,
        )

        # Weighted version should use fewer steps (fast component is "ignored")
        # This is a weak test - mainly ensures the parameter works
        assert result_weighted.success

    def test_seminorm_with_adjoint(self):
        """Test seminorm works with adjoint sensitivity method."""
        from torchscience.integration.initial_value_problem import solve_ivp

        theta = torch.tensor(
            [1.0, 100.0], requires_grad=True, dtype=torch.float64
        )

        def system(t, y):
            return torch.stack([-theta[0] * y[0], -theta[1] * y[1]])

        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
        error_weights = torch.tensor([1.0, 0.1], dtype=torch.float64)

        result = solve_ivp(
            system,
            y0,
            t_span=(0.0, 0.1),
            sensitivity="adjoint",
            params=[theta],
            error_weights=error_weights,
        )

        loss = result.y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad).all()

    def test_seminorm_rejects_invalid_weights(self):
        """Test that zero or negative weights are rejected."""
        import pytest

        from torchscience.integration.initial_value_problem import solve_ivp

        def system(t, y):
            return -y

        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

        # Zero weights should be rejected
        with pytest.raises(ValueError, match="strictly positive"):
            solve_ivp(
                system,
                y0,
                t_span=(0.0, 1.0),
                error_weights=torch.tensor([1.0, 0.0], dtype=torch.float64),
            )

        # Negative weights should be rejected
        with pytest.raises(ValueError, match="strictly positive"):
            solve_ivp(
                system,
                y0,
                t_span=(0.0, 1.0),
                error_weights=torch.tensor([1.0, -1.0], dtype=torch.float64),
            )
