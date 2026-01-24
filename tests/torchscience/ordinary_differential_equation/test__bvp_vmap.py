"""Tests for vmap compatibility."""

import pytest
import torch


class TestVmap:
    @pytest.mark.skip(
        reason="solve_bvp has dynamic control flow (Newton iterations, mesh adaptation) "
        "which is incompatible with torch.vmap. Use explicit loops for batched solving."
    )
    def test_vmap_solve_bvp(self):
        """Test vmap over solve_bvp - not supported due to dynamic control flow."""
        from torchscience.ordinary_differential_equation import solve_bvp

        def fun(x, y, p):
            return y

        def bc(ya, yb, p):
            return torch.stack(
                [ya[0] - 1.0, yb[0] - torch.exp(torch.tensor(1.0))]
            )

        def solve_single(y_init):
            x = torch.linspace(0, 1, 10, dtype=torch.float64)
            sol = solve_bvp(fun, bc, x, y_init.unsqueeze(0))
            return sol.y.squeeze(0)

        # Batch of initial guesses
        y_batch = torch.ones(5, 10, dtype=torch.float64)

        # vmap cannot work over solve_bvp due to:
        # 1. Newton iteration with data-dependent convergence
        # 2. Mesh adaptation that changes tensor sizes dynamically
        y_solutions = torch.vmap(solve_single)(y_batch)

        assert y_solutions.shape == (5, 10)

    def test_vmap_bvp_solution_call(self):
        """Test vmap over BVPSolution interpolation."""
        from torchscience.ordinary_differential_equation import BVPSolution

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.exp(x).unsqueeze(0)
        yp = torch.exp(x).unsqueeze(0)

        sol = BVPSolution(
            x=x,
            y=y,
            yp=yp,
            p=torch.empty(0),
            rms_residuals=torch.tensor(1e-4),
            n_iterations=5,
            success=True,
        )

        # vmap over query points batch
        x_batch = torch.stack(
            [
                torch.tensor([0.1, 0.2, 0.3]),
                torch.tensor([0.4, 0.5, 0.6]),
                torch.tensor([0.7, 0.8, 0.9]),
            ]
        )  # (3, 3)

        # This should work with vmap
        def interp_single(x_query):
            return sol(x_query)

        y_interp = torch.vmap(interp_single)(x_batch)
        assert y_interp.shape == (3, 1, 3)  # (batch, n_components, n_query)

    def test_hermite_interpolate_vmappable(self):
        """Test that hermite_interpolate is vmappable."""
        from torchscience.ordinary_differential_equation._interpolation import (
            hermite_interpolate,
        )

        x_nodes = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y_nodes = torch.tensor([[0.0, 0.25, 1.0]], dtype=torch.float64)
        yp_nodes = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)

        x_query = torch.tensor([0.25, 0.75], dtype=torch.float64)

        # Single call should work
        y_interp = hermite_interpolate(x_nodes, y_nodes, yp_nodes, x_query)
        assert y_interp.shape == (1, 2)

        # vmap over batch of y_nodes
        y_nodes_batch = torch.randn(5, 1, 3, dtype=torch.float64)
        yp_nodes_batch = torch.randn(5, 1, 3, dtype=torch.float64)

        def interp_fn(y, yp):
            return hermite_interpolate(x_nodes, y, yp, x_query)

        y_interp_batch = torch.vmap(interp_fn)(y_nodes_batch, yp_nodes_batch)
        assert y_interp_batch.shape == (5, 1, 2)

    def test_loop_based_batching_works(self):
        """Test that loop-based batching works as an alternative to vmap.

        This demonstrates the recommended approach for solving multiple BVPs
        when vmap is not available.
        """
        from torchscience.ordinary_differential_equation import solve_bvp

        def fun(x, y, p):
            return y

        def bc(ya, yb, p):
            e = torch.exp(torch.tensor(1.0, dtype=ya.dtype))
            return torch.stack([ya[0] - 1.0, yb[0] - e])

        x = torch.linspace(0, 1, 10, dtype=torch.float64)

        # Batch of initial guesses with slight perturbations
        batch_size = 5
        y_inits = [
            torch.ones(1, 10, dtype=torch.float64) * (1 + 0.1 * i)
            for i in range(batch_size)
        ]

        # Solve each BVP in a loop
        solutions = []
        for y_init in y_inits:
            sol = solve_bvp(fun, bc, x, y_init)
            solutions.append(sol)

        # All should converge to the same solution
        expected = torch.exp(solutions[0].x)
        for sol in solutions:
            assert sol.success
            torch.testing.assert_close(
                sol.y[0], expected, atol=1e-3, rtol=1e-3
            )
