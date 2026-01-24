"""Tests for BVP Newton solver."""

import torch

from torchscience.ordinary_differential_equation._bvp_newton import newton_bvp


class TestNewtonBVP:
    def test_linear_bvp(self):
        """Test Newton on y' = 1, y(0) = 0 (solution: y = x)."""

        def fun(x, y, p):
            return torch.ones_like(y)  # dy/dx = 1

        def bc(ya, yb, p):
            # y(0) = 0
            return ya  # Single component

        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = torch.zeros(1, 3, dtype=torch.float64)  # Initial guess
        p = torch.empty(0, dtype=torch.float64)

        y_sol, p_sol, converged, n_iter = newton_bvp(fun, bc, x, y, p)

        assert converged
        # Solution should be y = x
        expected = x.unsqueeze(0)
        torch.testing.assert_close(y_sol, expected, atol=1e-6, rtol=1e-6)

    def test_exponential_bvp(self):
        """Test Newton on y' = y, y(0) = 1 (solution: y = exp(x))."""

        def fun(x, y, p):
            return y  # dy/dx = y

        def bc(ya, yb, p):
            # y(0) = 1
            return ya - 1.0

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.ones(1, 5, dtype=torch.float64)  # Initial guess
        p = torch.empty(0, dtype=torch.float64)

        y_sol, p_sol, converged, n_iter = newton_bvp(fun, bc, x, y, p)

        assert converged
        # Solution should be close to exp(x)
        expected = torch.exp(x).unsqueeze(0)
        torch.testing.assert_close(y_sol, expected, atol=1e-3, rtol=1e-3)

    def test_second_order_ode(self):
        """Test Newton on y'' = 0, y(0) = 0, y(1) = 1 (solution: y = x)."""

        def fun(x, y, p):
            # y[0] = y, y[1] = y'
            # y' = y[1], y'' = 0
            return torch.stack([y[1], torch.zeros_like(y[0])])

        def bc(ya, yb, p):
            # y(0) = 0, y(1) = 1
            return torch.stack([ya[0], yb[0] - 1.0])

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.zeros(2, 5, dtype=torch.float64)
        y[1] = 1.0  # Initial guess for derivative
        p = torch.empty(0, dtype=torch.float64)

        y_sol, p_sol, converged, n_iter = newton_bvp(fun, bc, x, y, p)

        assert converged
        # y[0] should be x
        torch.testing.assert_close(y_sol[0], x, atol=1e-6, rtol=1e-6)

    def test_with_parameter(self):
        """Test Newton on y' = p*y, y(0) = 1, y(1) = e (solving for p=1)."""

        def fun(x, y, p):
            return p[0] * y

        def bc(ya, yb, p):
            e = torch.exp(torch.tensor(1.0, dtype=ya.dtype))
            return torch.stack([ya[0] - 1.0, yb[0] - e])

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.ones(1, 5, dtype=torch.float64)
        p = torch.tensor([0.5], dtype=torch.float64)  # Initial guess

        y_sol, p_sol, converged, n_iter = newton_bvp(fun, bc, x, y, p)

        assert converged
        # p should be close to 1
        torch.testing.assert_close(
            p_sol,
            torch.tensor([1.0], dtype=torch.float64),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_convergence_failure(self):
        """Test that Newton returns converged=False for difficult problems."""

        def fun(x, y, p):
            # Very nonlinear problem with poor initial guess
            return y**10

        def bc(ya, yb, p):
            return torch.stack([ya[0] - 1.0, yb[0] - 100.0])

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.zeros(1, 5, dtype=torch.float64)
        p = torch.empty(0, dtype=torch.float64)

        y_sol, p_sol, converged, n_iter = newton_bvp(
            fun, bc, x, y, p, max_iter=5
        )

        # Should not converge with such poor setup
        assert not converged
