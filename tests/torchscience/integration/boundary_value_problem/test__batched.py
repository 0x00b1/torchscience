"""Tests for batched BVP solving."""

import torch

from torchscience.integration.boundary_value_problem import solve_bvp


class TestBatchedBVP:
    def test_batched_linear_bvp(self):
        """Test solving multiple linear BVPs with different parameters.

        This test solves y' = a*y for different values of 'a' by looping
        over the parameter values. Each BVP has boundary conditions:
        y(0) = 1, y(1) = exp(a).

        The analytic solution is y = exp(a*x).
        """
        a_values = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
        x = torch.linspace(0, 1, 10, dtype=torch.float64)

        solutions = []
        for a in a_values:
            # Create closure over this specific 'a' value
            def make_fun(a_val):
                def fun(x, y, p):
                    return a_val * y

                return fun

            def make_bc(a_val):
                def bc(ya, yb, p):
                    return torch.stack([ya[0] - 1.0, yb[0] - torch.exp(a_val)])

                return bc

            fun = make_fun(a)
            bc = make_bc(a)
            y_init = torch.ones(1, 10, dtype=torch.float64)

            sol = solve_bvp(fun, bc, x, y_init)
            solutions.append(sol)

        # Verify each solution
        for i, sol in enumerate(solutions):
            assert sol.success
            expected = torch.exp(a_values[i] * sol.x)
            torch.testing.assert_close(
                sol.y[0], expected, atol=1e-2, rtol=1e-2
            )

    def test_batched_harmonic_oscillators(self):
        """Test solving multiple harmonic oscillator BVPs with different frequencies.

        Solves y'' + omega^2 * y = 0 for different omega values.
        With y(0) = 0, y(L) = sin(omega*L), the solution is y = sin(omega*x).
        """
        omega_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        L = torch.pi / 4  # Use pi/4 to avoid y(L)=0 for omega=2

        solutions = []
        for omega in omega_values:

            def make_fun(w):
                def fun(x, y, p):
                    return torch.stack([y[1], -(w**2) * y[0]])

                return fun

            def make_bc(w):
                def bc(ya, yb, p):
                    return torch.stack([ya[0], yb[0] - torch.sin(w * L)])

                return bc

            fun = make_fun(omega)
            bc = make_bc(omega)

            x = torch.linspace(0, L, 15, dtype=torch.float64)
            # Initial guess using sine
            y_init = torch.stack(
                [torch.sin(omega * x), omega * torch.cos(omega * x)]
            )

            sol = solve_bvp(fun, bc, x, y_init)
            solutions.append(sol)

        # Verify solutions
        for i, sol in enumerate(solutions):
            assert sol.success
            expected = torch.sin(omega_values[i] * sol.x)
            torch.testing.assert_close(
                sol.y[0], expected, atol=1e-3, rtol=1e-3
            )
