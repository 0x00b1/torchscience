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
