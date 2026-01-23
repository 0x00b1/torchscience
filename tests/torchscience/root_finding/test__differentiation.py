# tests/torchscience/root_finding/test__differentiation.py
"""Tests for differentiation utilities for root finding."""

import torch

from torchscience.root_finding._differentiation import (
    compute_derivative,
    compute_jacobian,
    compute_second_derivative,
)


class TestComputeDerivative:
    """Tests for compute_derivative function."""

    def test_autodiff_quadratic(self):
        """Autodiff derivative of f(x) = x^2 is 2x."""

        def f(x):
            return x**2

        x = torch.tensor([1.0, 2.0, 3.0])
        df = compute_derivative(f, x, method="autodiff")

        expected = 2 * x  # d/dx(x^2) = 2x
        torch.testing.assert_close(df, expected)

    def test_autodiff_sin(self):
        """Autodiff derivative of sin(x) is cos(x)."""

        def f(x):
            return torch.sin(x)

        x = torch.tensor([0.0, torch.pi / 4, torch.pi / 2])
        df = compute_derivative(f, x, method="autodiff")

        expected = torch.cos(x)
        torch.testing.assert_close(df, expected)

    def test_explicit_derivative(self):
        """When df is provided, use it directly."""

        def f(x):
            return x**3

        def df_func(x):
            return 3 * x**2

        x = torch.tensor([1.0, 2.0, 3.0])
        df = compute_derivative(f, x, df=df_func)

        expected = 3 * x**2  # d/dx(x^3) = 3x^2
        torch.testing.assert_close(df, expected)

    def test_finite_difference(self):
        """Finite difference approximation of derivative."""

        def f(x):
            return x**2

        # Use float64 for better finite difference precision
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        df = compute_derivative(f, x, method="finite_difference", h=1e-5)

        expected = 2 * x  # d/dx(x^2) = 2x
        torch.testing.assert_close(df, expected, rtol=1e-4, atol=1e-4)


class TestComputeSecondDerivative:
    """Tests for compute_second_derivative function."""

    def test_autodiff_quadratic(self):
        """Autodiff second derivative of f(x) = x^3 is 6x."""

        def f(x):
            return x**3

        x = torch.tensor([1.0, 2.0, 3.0])
        ddf = compute_second_derivative(f, x, method="autodiff")

        expected = 6 * x  # d^2/dx^2(x^3) = 6x
        torch.testing.assert_close(ddf, expected)

    def test_explicit_second_derivative(self):
        """When ddf is provided, use it directly."""

        def f(x):
            return x**4

        def ddf_func(x):
            return 12 * x**2

        x = torch.tensor([1.0, 2.0, 3.0])
        ddf = compute_second_derivative(f, x, ddf=ddf_func)

        expected = 12 * x**2  # d^2/dx^2(x^4) = 12x^2
        torch.testing.assert_close(ddf, expected)


class TestComputeJacobian:
    """Tests for compute_jacobian function."""

    def test_linear_system(self):
        """Jacobian of linear system Ax is A."""
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])

        def f(x):
            return x @ A.T

        x = torch.tensor([[1.0, 2.0]])  # (1, 2) batched input
        J = compute_jacobian(f, x)

        # Jacobian should be A (transposed due to how we defined f)
        expected = A.unsqueeze(0)  # (1, 2, 2)
        torch.testing.assert_close(J, expected)

    def test_nonlinear_system(self):
        """Jacobian of nonlinear system [x0^2 + x1, x0*x1]."""

        def f(x):
            # x is (B, 2), returns (B, 2)
            x0, x1 = x[..., 0], x[..., 1]
            f0 = x0**2 + x1
            f1 = x0 * x1
            return torch.stack([f0, f1], dim=-1)

        x = torch.tensor([[2.0, 3.0]])  # (1, 2)
        J = compute_jacobian(f, x)

        # Jacobian at (2, 3):
        # df0/dx0 = 2*x0 = 4, df0/dx1 = 1
        # df1/dx0 = x1 = 3, df1/dx1 = x0 = 2
        expected = torch.tensor([[[4.0, 1.0], [3.0, 2.0]]])  # (1, 2, 2)
        torch.testing.assert_close(J, expected)

    def test_explicit_jacobian(self):
        """When jacobian function is provided, use it directly."""

        def f(x):
            x0, x1 = x[..., 0], x[..., 1]
            return torch.stack([x0**2, x1**2], dim=-1)

        def jac_func(x):
            # Jacobian: [[2*x0, 0], [0, 2*x1]]
            B = x.shape[0]
            J = torch.zeros(B, 2, 2, dtype=x.dtype, device=x.device)
            J[:, 0, 0] = 2 * x[:, 0]
            J[:, 1, 1] = 2 * x[:, 1]
            return J

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        J = compute_jacobian(f, x, jacobian=jac_func)

        expected = torch.tensor(
            [[[2.0, 0.0], [0.0, 4.0]], [[6.0, 0.0], [0.0, 8.0]]]
        )  # (2, 2, 2)
        torch.testing.assert_close(J, expected)

    def test_batched_jacobian(self):
        """Jacobian computed for each batch element independently."""

        def f(x):
            # Simple linear function: f(x) = [2*x0 + x1, x0 - x1]
            x0, x1 = x[..., 0], x[..., 1]
            return torch.stack([2 * x0 + x1, x0 - x1], dim=-1)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        J = compute_jacobian(f, x)

        # Jacobian is constant: [[2, 1], [1, -1]]
        expected = (
            torch.tensor([[2.0, 1.0], [1.0, -1.0]])
            .unsqueeze(0)
            .expand(3, -1, -1)
        )
        torch.testing.assert_close(J, expected)
