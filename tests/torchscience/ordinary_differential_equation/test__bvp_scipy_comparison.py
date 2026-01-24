"""Tests comparing solve_bvp to scipy.integrate.solve_bvp."""

import numpy as np
import pytest
import torch

scipy_integrate = pytest.importorskip("scipy.integrate")


class TestScipyComparison:
    def test_bratu_problem(self):
        """Test Bratu equation: y'' + exp(y) = 0, y(0) = y(1) = 0."""
        from torchscience.ordinary_differential_equation import solve_bvp

        # PyTorch version
        def fun_torch(x, y, p):
            return torch.stack([y[1], -torch.exp(y[0])])

        def bc_torch(ya, yb, p):
            return torch.stack([ya[0], yb[0]])

        x_init = torch.linspace(0, 1, 10, dtype=torch.float64)
        y_init = torch.zeros(2, 10, dtype=torch.float64)

        sol_torch = solve_bvp(fun_torch, bc_torch, x_init, y_init)

        # SciPy version
        def fun_scipy(x, y):
            return np.array([y[1], -np.exp(y[0])])

        def bc_scipy(ya, yb):
            return np.array([ya[0], yb[0]])

        x_scipy = np.linspace(0, 1, 10)
        y_scipy = np.zeros((2, 10))

        sol_scipy = scipy_integrate.solve_bvp(
            fun_scipy, bc_scipy, x_scipy, y_scipy
        )

        # Compare solutions at common points
        x_compare = torch.linspace(0, 1, 50, dtype=torch.float64)
        y_torch = sol_torch(x_compare)
        y_scipy_interp = sol_scipy.sol(x_compare.numpy())

        torch.testing.assert_close(
            y_torch[0],
            torch.from_numpy(y_scipy_interp[0]),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_harmonic_oscillator(self):
        """Test y'' + y = 0, y(0) = 0, y(pi/2) = 1 (solution: sin(x))."""
        from torchscience.ordinary_differential_equation import solve_bvp

        def fun(x, y, p):
            return torch.stack([y[1], -y[0]])

        def bc(ya, yb, p):
            return torch.stack([ya[0], yb[0] - 1.0])

        x = torch.linspace(0, torch.pi / 2, 10, dtype=torch.float64)
        y = torch.stack([torch.sin(x), torch.cos(x)])

        sol = solve_bvp(fun, bc, x, y)

        assert sol.success
        expected = torch.sin(sol.x)
        torch.testing.assert_close(sol.y[0], expected, atol=1e-3, rtol=1e-3)

    def test_nonlinear_stiff(self):
        """Test nonlinear problem: y' = y^2, y(0) = 1, y(0.5) = 2."""
        from torchscience.ordinary_differential_equation import solve_bvp

        # Solution: y = 1/(1-x)

        def fun(x, y, p):
            return y**2

        def bc(ya, yb, p):
            return torch.stack([ya[0] - 1.0, yb[0] - 2.0])

        x = torch.linspace(0, 0.5, 10, dtype=torch.float64)
        y = torch.ones(1, 10, dtype=torch.float64)

        sol = solve_bvp(fun, bc, x, y)

        assert sol.success
        expected = 1.0 / (1.0 - sol.x)
        torch.testing.assert_close(sol.y[0], expected, atol=1e-3, rtol=1e-3)
