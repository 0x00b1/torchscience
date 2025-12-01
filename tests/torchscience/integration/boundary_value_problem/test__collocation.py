"""Tests for collocation infrastructure."""

import torch

from torchscience.integration.boundary_value_problem._collocation import (
    compute_collocation_residual,
    get_lobatto_coefficients,
)


class TestLobattoCoefficients:
    def test_coefficients_shape(self):
        """Test Lobatto coefficients have correct shapes."""
        c, A, B = get_lobatto_coefficients("float64", "cpu")

        # 3-point Lobatto IIIA: nodes at 0, 0.5, 1
        assert c.shape == (3,)
        assert A.shape == (3, 3)
        assert B.shape == (3,)

    def test_nodes_are_correct(self):
        """Test Lobatto nodes are [0, 0.5, 1]."""
        c, A, B = get_lobatto_coefficients("float64", "cpu")

        expected_c = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        torch.testing.assert_close(c, expected_c)

    def test_coefficients_sum_to_weights(self):
        """Test each row of A sums to c (RK consistency)."""
        c, A, B = get_lobatto_coefficients("float64", "cpu")

        row_sums = A.sum(dim=1)
        torch.testing.assert_close(row_sums, c)

    def test_weights_sum_to_one(self):
        """Test quadrature weights sum to 1 (for unit interval)."""
        c, A, B = get_lobatto_coefficients("float64", "cpu")

        torch.testing.assert_close(
            B.sum(), torch.tensor(1.0, dtype=torch.float64)
        )

    def test_different_dtypes(self):
        """Test coefficients work with float32 and float64."""
        for dtype_str in ["float32", "float64"]:
            c, A, B = get_lobatto_coefficients(dtype_str, "cpu")
            expected_dtype = getattr(torch, dtype_str)
            assert c.dtype == expected_dtype
            assert A.dtype == expected_dtype
            assert B.dtype == expected_dtype


class TestCollocationResidual:
    def test_exponential_solution_small_residual(self):
        """Test that exact solution of y' = y has small (4th-order) residual."""
        # For y' = y, the exact solution is y = exp(x)
        # Cubic Hermite is only 4th-order accurate, so residual is O(h^4)

        def fun(x, y, p):
            return y  # dy/dx = y

        x = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = torch.exp(x).unsqueeze(0)  # shape (1, 2)
        p = torch.empty(0, dtype=torch.float64)

        residual = compute_collocation_residual(fun, x, y, p)

        # Residual should be small but not zero (4th-order error)
        assert residual.shape == (1, 1)  # (n_components, n_intervals)
        # For h=1, 4th-order error is roughly O(1) for exp, but should be modest
        assert residual.abs().max() < 0.01  # Much smaller than 1

    def test_residual_shape(self):
        """Test residual has correct shape."""

        def fun(x, y, p):
            return torch.stack([y[1], -y[0]])  # Harmonic oscillator

        x = torch.linspace(0, 1, 5, dtype=torch.float64)  # 4 intervals
        y = torch.randn(2, 5, dtype=torch.float64)
        p = torch.empty(0, dtype=torch.float64)

        residual = compute_collocation_residual(fun, x, y, p)

        # (n_components, n_intervals) = (2, 4)
        assert residual.shape == (2, 4)

    def test_linear_ode_exact(self):
        """Test y' = 1 with y(x) = x gives zero residual."""

        def fun(x, y, p):
            return torch.ones_like(y)  # dy/dx = 1

        x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = x.unsqueeze(0)  # y = x, shape (1, 3)
        p = torch.empty(0, dtype=torch.float64)

        residual = compute_collocation_residual(fun, x, y, p)

        torch.testing.assert_close(
            residual, torch.zeros_like(residual), atol=1e-10, rtol=1e-10
        )

    def test_vectorized_minimal_calls(self):
        """Test that fun is called only twice (nodes + midpoints), not per-interval."""
        call_count = [0]

        def fun(x, y, p):
            call_count[0] += 1
            return y

        x = torch.linspace(0, 1, 10, dtype=torch.float64)  # 9 intervals
        y = torch.randn(2, 10, dtype=torch.float64)
        p = torch.empty(0, dtype=torch.float64)

        compute_collocation_residual(fun, x, y, p)

        # Should call fun twice:
        # 1. Once for all mesh nodes (10 points)
        # 2. Once for all midpoints (9 points)
        # NOT once per interval (which would be 9 calls)
        assert call_count[0] == 2
