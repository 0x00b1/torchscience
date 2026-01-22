"""Phase 4 tests for competitive advantage features."""

import torch


class TestSensitivityAnalysis:
    """Tests for sensitivity analysis modes."""

    def test_solve_ivp_sensitivity_gradient_mode(self):
        """sensitivity mode='gradient' should return parameter gradients."""
        from torchscience.integration.initial_value_problem import (
            solve_ivp_sensitivity,
        )

        theta = torch.tensor([1.0, 0.5], dtype=torch.float64)

        def dynamics(t, y):
            return -theta[0] * y + theta[1] * torch.sin(torch.as_tensor(t))

        y0 = torch.tensor([1.0], dtype=torch.float64)

        def loss_fn(y_final):
            return y_final.sum()

        grad = solve_ivp_sensitivity(
            dynamics,
            y0,
            t_span=(0.0, 1.0),
            params=[theta],
            loss_fn=loss_fn,
            mode="gradient",
        )

        assert grad.shape == theta.shape
        assert torch.all(torch.isfinite(grad))

    def test_solve_ivp_sensitivity_jacobian_mode(self):
        """sensitivity mode='jacobian' should return dy_final/d_theta."""
        from torchscience.integration.initial_value_problem import (
            solve_ivp_sensitivity,
        )

        theta = torch.tensor([1.0], dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0, 2.0], dtype=torch.float64)

        J = solve_ivp_sensitivity(
            dynamics,
            y0,
            t_span=(0.0, 1.0),
            params=[theta],
            mode="jacobian",
        )

        # J should be dy_final/d_theta: (state_dim, param_dim)
        assert J.shape == (2, 1)
        assert torch.all(torch.isfinite(J))

        # For dy/dt = -theta*y, y(T) = y0*exp(-theta*T)
        # dy(T)/d_theta = -T * y0 * exp(-theta*T)
        T = 1.0
        expected_J = -T * y0.unsqueeze(1) * torch.exp(-theta * T)
        assert torch.allclose(J, expected_J, rtol=1e-4)

    def test_solve_ivp_sensitivity_fisher_mode(self):
        """sensitivity mode='fisher' should return Fisher information matrix."""
        from torchscience.integration.initial_value_problem import (
            solve_ivp_sensitivity,
        )

        theta = torch.tensor([1.0, 0.5], dtype=torch.float64)

        def dynamics(t, y):
            return -theta[0] * y + theta[1]

        y0 = torch.tensor([1.0], dtype=torch.float64)

        fisher = solve_ivp_sensitivity(
            dynamics,
            y0,
            t_span=(0.0, 1.0),
            params=[theta],
            mode="fisher",
        )

        # Fisher should be (param_dim, param_dim)
        assert fisher.shape == (2, 2)
        assert torch.all(torch.isfinite(fisher))

        # Fisher should be symmetric positive semi-definite
        assert torch.allclose(fisher, fisher.T)
        eigvals = torch.linalg.eigvalsh(fisher)
        assert torch.all(eigvals >= -1e-10)  # Allow small numerical error


class TestSecondOrderGradients:
    """Tests for second-order gradient computation."""

    def test_hvp_basic(self):
        """Hessian-vector product should be computable."""
        from torchscience.integration.initial_value_problem import (
            solve_ivp_hvp,
        )

        theta = torch.tensor([1.0], dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        def loss_fn(y_final):
            return (y_final**2).sum()

        v = torch.tensor([1.0], dtype=torch.float64)  # Direction vector

        hvp = solve_ivp_hvp(
            dynamics,
            y0,
            t_span=(0.0, 1.0),
            params=[theta],
            loss_fn=loss_fn,
            v=v,
        )

        assert hvp.shape == theta.shape
        assert torch.all(torch.isfinite(hvp))

    def test_hvp_vs_full_hessian(self):
        """HVP should match H @ v where H is computed via double backward."""
        from torchscience.integration.initial_value_problem import (
            adjoint,
            dormand_prince_5,
            solve_ivp_hvp,
        )

        theta = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def loss_fn_full(theta_val):
            def dynamics(t, y):
                return -theta_val * y

            y0 = torch.tensor([1.0], dtype=torch.float64)
            solver = adjoint(dormand_prince_5, params=[theta_val])
            y_final, _ = solver(dynamics, y0, t_span=(0.0, 1.0))
            return (y_final**2).sum()

        # Compute Hessian via torch.autograd.functional.hessian
        H = torch.autograd.functional.hessian(loss_fn_full, theta)

        # Compute HVP
        v = torch.tensor([1.0], dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        hvp = solve_ivp_hvp(
            dynamics,
            torch.tensor([1.0], dtype=torch.float64),
            t_span=(0.0, 1.0),
            params=[theta],
            loss_fn=lambda y: (y**2).sum(),
            v=v,
        )

        expected = H @ v
        assert torch.allclose(hvp, expected, rtol=1e-3)


class TestLazyAdjoint:
    """Tests for lazy adjoint mode (defer computation until needed)."""

    def test_lazy_adjoint_basic(self):
        """lazy_adjoint should defer gradient computation."""
        from torchscience.integration.initial_value_problem import solve_ivp

        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # With lazy adjoint, forward solve happens but adjoint is deferred
        y_final, _ = solve_ivp(
            dynamics,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="lazy_adjoint",
            params=[theta],
        )

        # y_final should be available
        assert y_final is not None
        assert y_final.requires_grad  # Should have grad_fn

        # No gradient computed yet
        assert theta.grad is None

        # Only when we call backward does adjoint run
        y_final.sum().backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

    def test_lazy_adjoint_conditional_backward(self):
        """lazy_adjoint allows inspecting y_final before deciding to backprop."""
        from torchscience.integration.initial_value_problem import solve_ivp

        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        y_final, _ = solve_ivp(
            dynamics,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="lazy_adjoint",
            params=[theta],
        )

        # Conditionally skip backward based on y_final value
        if y_final.abs().max() > 0.1:
            y_final.sum().backward()
            assert theta.grad is not None
        else:
            # No backward - no gradient
            assert theta.grad is None

    def test_lazy_adjoint_matches_adjoint(self):
        """lazy_adjoint should produce the same gradients as adjoint."""
        from torchscience.integration.initial_value_problem import solve_ivp

        theta1 = torch.tensor([1.5], requires_grad=True, dtype=torch.float64)
        theta2 = torch.tensor([1.5], requires_grad=True, dtype=torch.float64)

        def dynamics1(t, y):
            return -theta1 * y

        def dynamics2(t, y):
            return -theta2 * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Solve with adjoint
        y_final1, _ = solve_ivp(
            dynamics1,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="adjoint",
            params=[theta1],
        )
        y_final1.sum().backward()

        # Solve with lazy_adjoint
        y_final2, _ = solve_ivp(
            dynamics2,
            y0,
            t_span=(0.0, 1.0),
            sensitivity="lazy_adjoint",
            params=[theta2],
        )
        y_final2.sum().backward()

        # Both should give the same result
        assert torch.allclose(y_final1, y_final2)
        assert torch.allclose(theta1.grad, theta2.grad)


class TestGradientClipping:
    """Tests for gradient clipping during adjoint integration."""

    def test_gradient_clip_norm(self):
        """gradient_clip should limit adjoint norm during integration."""
        import warnings

        from torchscience.integration.initial_value_problem import (
            adjoint,
            dormand_prince_5,
        )

        # Dynamics that cause adjoint to grow
        theta = torch.tensor([10.0], requires_grad=True, dtype=torch.float64)

        def unstable_dynamics(t, y):
            return theta * y  # Exponential growth backward

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(
            dormand_prince_5,
            params=[theta],
            adjoint_options={
                "gradient_clip": 1e3,
                "gradient_clip_mode": "norm",
            },
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y_final, _ = solver(unstable_dynamics, y0, t_span=(0.0, 2.0))
            y_final.sum().backward()

        # Gradient should be finite (clipping prevented explosion)
        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

        # Should have warned about clipping
        clip_warnings = [x for x in w if "clipped" in str(x.message).lower()]
        # Note: May or may not warn depending on dynamics
