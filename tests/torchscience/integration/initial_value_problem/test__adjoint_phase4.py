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
