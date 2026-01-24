"""Tests for BVPSolution tensorclass."""

import torch

from torchscience.integration._bvp_solution import (
    BVPSolution,
)


class TestBVPSolution:
    def test_creation(self):
        """Test BVPSolution can be created with required fields."""
        x = torch.linspace(0, 1, 10)
        y = torch.randn(2, 10)
        yp = torch.randn(2, 10)  # derivatives
        p = torch.tensor([1.0])

        sol = BVPSolution(
            x=x,
            y=y,
            yp=yp,
            p=p,
            rms_residuals=torch.tensor(1e-4),
            n_iterations=5,
            success=True,
        )

        assert sol.x.shape == (10,)
        assert sol.y.shape == (2, 10)
        assert sol.yp.shape == (2, 10)
        assert sol.p.shape == (1,)
        assert sol.success.item() is True

    def test_call_interpolates_at_nodes(self):
        """Test BVPSolution returns exact values at mesh points."""
        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = torch.tensor(
            [[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]], dtype=torch.float64
        )
        yp = torch.tensor(
            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=torch.float64
        )

        sol = BVPSolution(
            x=x,
            y=y,
            yp=yp,
            p=torch.empty(0, dtype=torch.float64),
            rms_residuals=torch.tensor(1e-4),
            n_iterations=5,
            success=True,
        )

        # Query at mesh points should return exact values
        y_query = sol(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
        assert y_query.shape == (2, 3)
        torch.testing.assert_close(y_query, y)

    def test_cubic_hermite_exact_for_cubic(self):
        """Test cubic Hermite is exact for cubic polynomials."""
        # y = x^3, y' = 3x^2
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = x.unsqueeze(0) ** 3  # [[0, 1]]
        yp = 3 * x.unsqueeze(0) ** 2  # [[0, 3]]

        sol = BVPSolution(
            x=x,
            y=y,
            yp=yp,
            p=torch.empty(0, dtype=torch.float64),
            rms_residuals=torch.tensor(1e-4),
            n_iterations=5,
            success=True,
        )

        # Cubic Hermite should be exact for cubic polynomials
        x_query = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        y_query = sol(x_query)
        expected = x_query.unsqueeze(0) ** 3
        torch.testing.assert_close(y_query, expected, atol=1e-10, rtol=1e-10)

    def test_interpolation_quadratic(self):
        """Test interpolation for y = x^2 (should be near-exact)."""
        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = x.unsqueeze(0) ** 2  # [[0, 1]]
        yp = 2 * x.unsqueeze(0)  # [[0, 2]]

        sol = BVPSolution(
            x=x,
            y=y,
            yp=yp,
            p=torch.empty(0, dtype=torch.float64),
            rms_residuals=torch.tensor(1e-4),
            n_iterations=5,
            success=True,
        )

        y_query = sol(torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
        expected = torch.tensor([[0.0625, 0.25, 0.5625]], dtype=torch.float64)
        torch.testing.assert_close(y_query, expected, atol=1e-10, rtol=1e-10)
