"""Tests for tridiagonal solver."""

import torch


class TestSolveTridiagonal:
    def test_simple_3x3_system(self):
        """Test solving a simple 3x3 tridiagonal system."""
        from torchscience.spline._solve_tridiagonal import solve_tridiagonal

        # System: [2 1 0] [x0]   [1]
        #         [1 2 1] [x1] = [2]
        #         [0 1 2] [x2]   [1]
        diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        upper = torch.tensor([1.0, 1.0], dtype=torch.float64)
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64)
        rhs = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)

        x = solve_tridiagonal(diag, upper, lower, rhs)

        # Verify Ax = b
        result = torch.zeros_like(rhs)
        result[0] = diag[0] * x[0] + upper[0] * x[1]
        result[1] = lower[0] * x[0] + diag[1] * x[1] + upper[1] * x[2]
        result[2] = lower[1] * x[1] + diag[2] * x[2]

        torch.testing.assert_close(result, rhs, rtol=1e-10, atol=1e-10)

    def test_batched_rhs(self):
        """Test with batched right-hand side."""
        from torchscience.spline._solve_tridiagonal import solve_tridiagonal

        diag = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)
        upper = torch.tensor([1.0, 1.0], dtype=torch.float64)
        lower = torch.tensor([1.0, 1.0], dtype=torch.float64)
        # Batch of 2 right-hand sides, each of length 3
        rhs = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0]], dtype=torch.float64
        )

        x = solve_tridiagonal(diag, upper, lower, rhs)

        assert x.shape == (2, 3)
        # Second solution should be 2x the first
        torch.testing.assert_close(x[1], 2 * x[0], rtol=1e-10, atol=1e-10)

    def test_gradcheck(self):
        """Test gradients through the solver."""
        from torchscience.spline._solve_tridiagonal import solve_tridiagonal

        diag = torch.tensor(
            [2.0, 2.0, 2.0], dtype=torch.float64, requires_grad=True
        )
        upper = torch.tensor(
            [1.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        lower = torch.tensor(
            [1.0, 1.0], dtype=torch.float64, requires_grad=True
        )
        rhs = torch.tensor(
            [1.0, 2.0, 1.0], dtype=torch.float64, requires_grad=True
        )

        def fn(d, u, l, r):
            return solve_tridiagonal(d, u, l, r)

        assert torch.autograd.gradcheck(
            fn, (diag, upper, lower, rhs), eps=1e-6
        )

    def test_larger_system(self):
        """Test a larger system against torch.linalg.solve."""
        from torchscience.spline._solve_tridiagonal import solve_tridiagonal

        n = 50
        diag = 4 * torch.ones(n, dtype=torch.float64)
        upper = torch.ones(n - 1, dtype=torch.float64)
        lower = torch.ones(n - 1, dtype=torch.float64)
        rhs = torch.randn(n, dtype=torch.float64)

        x = solve_tridiagonal(diag, upper, lower, rhs)

        # Build full matrix and solve with torch.linalg.solve
        A = torch.diag(diag) + torch.diag(upper, 1) + torch.diag(lower, -1)
        x_expected = torch.linalg.solve(A, rhs)

        torch.testing.assert_close(x, x_expected, rtol=1e-10, atol=1e-10)
