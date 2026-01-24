# tests/torchscience/integration/initial_value_problem/test__adjoint_accuracy.py
"""Tests verifying adjoint gradient accuracy against finite differences."""

import pytest
import torch

from torchscience.integration import (
    adjoint,
    dormand_prince_5,
    euler,
    runge_kutta_4,
)


class TestAdjointGradientAccuracy:
    @pytest.mark.parametrize(
        "solver,kwargs",
        [
            (euler, {"dt": 0.01}),
            (runge_kutta_4, {"dt": 0.01}),
            (dormand_prince_5, {"rtol": 1e-6, "atol": 1e-9}),
        ],
    )
    def test_gradient_vs_finite_diff(self, solver, kwargs):
        """Compare adjoint gradients to finite difference approximation."""
        eps = 1e-5

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Adjoint gradient
        theta = torch.tensor([1.5], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -theta * y

        adjoint_solver = adjoint(solver)
        y_final, _ = adjoint_solver(f, y0.clone(), t_span=(0.0, 1.0), **kwargs)
        loss = y_final.sum()
        loss.backward()
        grad_adjoint = theta.grad.clone()

        # Finite difference
        def compute_loss(theta_val):
            def f_local(t, y):
                return -theta_val * y

            y_final, _ = solver(
                f_local, y0.clone(), t_span=(0.0, 1.0), **kwargs
            )
            return y_final.sum()

        with torch.no_grad():
            theta_plus = torch.tensor([1.5 + eps], dtype=torch.float64)
            theta_minus = torch.tensor([1.5 - eps], dtype=torch.float64)
            loss_plus = compute_loss(theta_plus)
            loss_minus = compute_loss(theta_minus)
            grad_fd = (loss_plus - loss_minus) / (2 * eps)

        # Should match within reasonable tolerance
        # (adjoint is approximate due to discretization of adjoint ODE)
        assert torch.allclose(grad_adjoint, grad_fd, rtol=0.1, atol=1e-4)

    def test_gradient_multiple_params(self):
        """Test gradients with multiple learnable parameters."""
        alpha = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
        beta = torch.tensor([0.5], requires_grad=True, dtype=torch.float64)

        def f(t, y):
            return -alpha * y + beta * torch.sin(
                torch.tensor(t, dtype=torch.float64)
            )

        y0 = torch.tensor([1.0], dtype=torch.float64)

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.01)
        loss = y_final.sum()
        loss.backward()

        assert alpha.grad is not None
        assert beta.grad is not None
        assert not torch.isnan(alpha.grad).any()
        assert not torch.isnan(beta.grad).any()

    def test_gradient_2d_param(self):
        """Test with matrix-valued parameter."""
        A = torch.tensor([[-1.0, 0.5], [0.5, -1.0]], requires_grad=True)

        def f(t, y):
            return A @ y

        y0 = torch.tensor([1.0, 0.5])

        adjoint_solver = adjoint(runge_kutta_4)
        y_final, _ = adjoint_solver(f, y0, t_span=(0.0, 1.0), dt=0.01)
        loss = y_final.sum()
        loss.backward()

        assert A.grad is not None
        assert A.grad.shape == A.shape


class TestAdjointVsDirectBackprop:
    def test_comparison_exponential_decay(self):
        """Compare adjoint to direct backprop in detail."""

        def make_dynamics(theta):
            def f(t, y):
                return -theta * y

            return f

        y0 = torch.tensor([1.0], dtype=torch.float64)

        # Direct backprop
        theta_direct = torch.tensor(
            [1.0], requires_grad=True, dtype=torch.float64
        )
        f_direct = make_dynamics(theta_direct)
        y_direct, _ = runge_kutta_4(
            f_direct, y0.clone(), t_span=(0.0, 1.0), dt=0.01
        )
        loss_direct = y_direct.sum()
        loss_direct.backward()
        grad_direct = theta_direct.grad.clone()

        # Adjoint
        theta_adjoint = torch.tensor(
            [1.0], requires_grad=True, dtype=torch.float64
        )
        f_adjoint = make_dynamics(theta_adjoint)
        adjoint_solver = adjoint(runge_kutta_4)
        y_adjoint, _ = adjoint_solver(
            f_adjoint, y0.clone(), t_span=(0.0, 1.0), dt=0.01
        )
        loss_adjoint = y_adjoint.sum()
        loss_adjoint.backward()
        grad_adjoint = theta_adjoint.grad.clone()

        # Compare
        print(f"Direct grad: {grad_direct.item():.6f}")
        print(f"Adjoint grad: {grad_adjoint.item():.6f}")
        rel_diff = (
            (grad_adjoint - grad_direct).abs() / grad_direct.abs()
        ).item()
        print(f"Relative diff: {rel_diff:.4f}")

        # Should be reasonably close
        assert torch.allclose(grad_direct, grad_adjoint, rtol=0.2)
