"""Tests for meta tensor support."""

import torch

from torchscience.integration.boundary_value_problem import solve_bvp


class TestMetaTensor:
    def test_solve_bvp_meta_output_shape(self):
        """Test solve_bvp returns correct shapes with meta tensors."""

        def fun(x, y, p):
            return torch.stack([y[1], -y[0]])

        def bc(ya, yb, p):
            return torch.stack([ya[0], yb[0] - 1.0])

        # Use meta tensors
        x = torch.linspace(0, 1, 10, dtype=torch.float64, device="meta")
        y = torch.zeros(2, 10, dtype=torch.float64, device="meta")

        sol = solve_bvp(fun, bc, x, y)

        # Should preserve meta device and have correct shapes
        assert sol.x.device.type == "meta"
        assert sol.y.device.type == "meta"
        assert sol.x.shape[0] >= 10  # At least as many nodes as input
        assert sol.y.shape == (2, sol.x.shape[0])

    def test_bvp_solution_interpolation_meta(self):
        """Test BVPSolution interpolation with meta tensors."""
        from torchscience.integration.boundary_value_problem import BVPSolution

        x = torch.linspace(0, 1, 10, device="meta")
        y = torch.zeros(2, 10, device="meta")
        yp = torch.zeros(2, 10, device="meta")
        p = torch.empty(0, device="meta")

        sol = BVPSolution(
            x=x,
            y=y,
            yp=yp,
            p=p,
            rms_residuals=torch.tensor(1e-4, device="meta"),
            n_iterations=5,
            success=True,
        )

        x_query = torch.tensor([0.25, 0.5, 0.75], device="meta")
        y_query = sol(x_query)

        assert y_query.device.type == "meta"
        assert y_query.shape == (2, 3)
