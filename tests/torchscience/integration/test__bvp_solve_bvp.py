"""Tests for solve_bvp main API."""

import torch

from torchscience.integration import (
    BVPSolution,
    solve_bvp,
)


class TestSolveBVP:
    def test_linear_bvp(self):
        """Test solving y'' = 0 with y(0) = 0, y(1) = 1."""

        # Convert to first-order system: y0' = y1, y1' = 0
        def fun(x, y, p):
            return torch.stack([y[1], torch.zeros_like(y[0])])

        def bc(ya, yb, p):
            return torch.stack([ya[0], yb[0] - 1.0])

        x = torch.linspace(0, 1, 5, dtype=torch.float64)
        y = torch.zeros(2, 5, dtype=torch.float64)

        sol = solve_bvp(fun, bc, x, y)

        assert isinstance(sol, BVPSolution)
        assert sol.success
        # Solution should be y = x
        expected = sol.x
        torch.testing.assert_close(sol.y[0], expected, atol=1e-4, rtol=1e-4)

    def test_exponential_bvp(self):
        """Test solving y' = y with y(0) = 1."""

        def fun(x, y, p):
            return y

        def bc(ya, yb, p):
            # y(0) = 1, y(1) = e
            e = torch.exp(torch.tensor(1.0, dtype=ya.dtype))
            return torch.stack([ya[0] - 1.0, yb[0] - e])

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.ones(1, 10, dtype=torch.float64)

        sol = solve_bvp(fun, bc, x, y)

        assert sol.success
        expected = torch.exp(sol.x)
        torch.testing.assert_close(sol.y[0], expected, atol=1e-3, rtol=1e-3)

    def test_unknown_parameter(self):
        """Test solving BVP with unknown parameter (eigenvalue problem)."""
        # y'' + p*y = 0, y(0) = 0, y(pi) = 0, y'(0) = 1
        # Solution: y = sin(sqrt(p)*x), eigenvalue p = 1

        def fun(x, y, p):
            return torch.stack([y[1], -p[0] * y[0]])

        def bc(ya, yb, p):
            return torch.stack([ya[0], yb[0], ya[1] - 1.0])

        x = torch.linspace(0, torch.pi, 20, dtype=torch.float64)
        y = torch.stack(
            [
                torch.sin(x),  # Initial guess
                torch.cos(x),
            ]
        )
        p = torch.tensor([1.0], dtype=torch.float64)  # Initial guess

        sol = solve_bvp(fun, bc, x, y, p=p)

        assert sol.success
        # Parameter should be close to 1
        torch.testing.assert_close(
            sol.p,
            torch.tensor([1.0], dtype=torch.float64),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_interpolation(self):
        """Test that solution is callable for interpolation."""

        def fun(x, y, p):
            return y

        def bc(ya, yb, p):
            e = torch.exp(torch.tensor(1.0, dtype=ya.dtype))
            return torch.stack([ya[0] - 1.0, yb[0] - e])

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.ones(1, 10, dtype=torch.float64)

        sol = solve_bvp(fun, bc, x, y)

        # Evaluate at arbitrary points
        x_query = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        y_query = sol(x_query)

        expected = torch.exp(x_query)
        torch.testing.assert_close(y_query[0], expected, atol=1e-2, rtol=1e-2)
