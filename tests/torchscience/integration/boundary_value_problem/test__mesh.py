"""Tests for mesh adaptation."""

import torch

from torchscience.integration.boundary_value_problem._mesh import (
    compute_rms_residuals,
    refine_mesh,
)


class TestComputeRMSResiduals:
    def test_rms_shape(self):
        """Test RMS residuals has correct shape."""
        residual = torch.randn(
            2, 5, dtype=torch.float64
        )  # 2 components, 5 intervals
        rms = compute_rms_residuals(residual)
        assert rms.shape == (5,)

    def test_rms_values(self):
        """Test RMS computation is correct."""
        # Single component case: RMS = abs(residual)
        residual = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        rms = compute_rms_residuals(residual)
        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(rms, expected)

        # Two component case: RMS = sqrt((r1^2 + r2^2) / 2)
        residual = torch.tensor([[3.0, 0.0], [4.0, 5.0]], dtype=torch.float64)
        rms = compute_rms_residuals(residual)
        expected = torch.tensor(
            [
                torch.sqrt(torch.tensor(25.0 / 2)),
                torch.sqrt(torch.tensor(25.0 / 2)),
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(rms, expected)


class TestRefineMesh:
    def test_refine_uniform_mesh(self):
        """Test mesh refinement on uniform mesh with high residual."""
        x = torch.linspace(0, 1, 3, dtype=torch.float64)  # 2 intervals
        y = torch.zeros(1, 3, dtype=torch.float64)
        f = torch.ones(1, 3, dtype=torch.float64)

        # High residuals everywhere -> should refine
        rms_res = torch.tensor([1.0, 1.0], dtype=torch.float64)

        x_new, y_new = refine_mesh(x, y, f, rms_res, tol=0.1)

        # Should have more nodes
        assert x_new.shape[0] > x.shape[0]
        # Should preserve endpoints
        torch.testing.assert_close(x_new[0], x[0])
        torch.testing.assert_close(x_new[-1], x[-1])

    def test_no_refine_when_converged(self):
        """Test mesh is not refined when residuals are below tolerance."""
        x = torch.linspace(0, 1, 5, dtype=torch.float64)  # 4 intervals
        y = torch.zeros(1, 5, dtype=torch.float64)
        f = torch.zeros(1, 5, dtype=torch.float64)

        # Low residuals everywhere -> no refinement needed
        rms_res = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6], dtype=torch.float64)

        x_new, y_new = refine_mesh(x, y, f, rms_res, tol=1e-3)

        # Should not add nodes
        assert x_new.shape[0] == x.shape[0]
        torch.testing.assert_close(x_new, x)

    def test_selective_refinement(self):
        """Test only intervals with high residuals are refined."""
        x = torch.linspace(0, 1, 5, dtype=torch.float64)  # 4 intervals
        y = torch.zeros(1, 5, dtype=torch.float64)
        f = torch.zeros(1, 5, dtype=torch.float64)

        # Only first interval has high residual
        rms_res = torch.tensor([1.0, 1e-6, 1e-6, 1e-6], dtype=torch.float64)

        x_new, y_new = refine_mesh(x, y, f, rms_res, tol=0.1)

        # Should add node in first interval only
        assert x_new.shape[0] == 6  # 5 + 1

    def test_preserves_monotonicity(self):
        """Test refined mesh is strictly increasing."""
        x = torch.tensor([0.0, 0.3, 0.7, 1.0], dtype=torch.float64)
        y = torch.randn(2, 4, dtype=torch.float64)
        f = torch.randn(2, 4, dtype=torch.float64)
        rms_res = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

        x_new, y_new = refine_mesh(x, y, f, rms_res, tol=0.1)

        # Check strictly increasing
        diffs = x_new[1:] - x_new[:-1]
        assert (diffs > 0).all()
