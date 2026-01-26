import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.linear._quadratic_program import (
    quadratic_program,
)


class TestQuadraticProgram:
    def test_unconstrained(self):
        """min 0.5 x^T Q x + c^T x with Q = I, c = [-1, -2]."""
        Q = torch.eye(2)
        c = torch.tensor([-1.0, -2.0])
        result = quadratic_program(Q, c)
        # Solution: x = Q^{-1} (-c) = [1, 2]
        torch.testing.assert_close(
            result.x,
            torch.tensor([1.0, 2.0]),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_equality_constraint(self):
        """min 0.5 x^T I x subject to x1 + x2 = 1."""
        Q = torch.eye(2)
        c = torch.zeros(2)
        A_eq = torch.tensor([[1.0, 1.0]])
        b_eq = torch.tensor([1.0])
        result = quadratic_program(Q, c, A_eq=A_eq, b_eq=b_eq)
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result.x, expected, atol=1e-3, rtol=1e-3)

    def test_inequality_constraint(self):
        """min 0.5 x^T I x - [2, 2]^T x subject to x1 + x2 <= 1."""
        Q = torch.eye(2)
        c = torch.tensor([-2.0, -2.0])
        A_ub = torch.tensor([[1.0, 1.0]])
        b_ub = torch.tensor([1.0])
        result = quadratic_program(Q, c, A_ub=A_ub, b_ub=b_ub)
        # Unconstrained optimum is [2, 2], but constraint pushes to [0.5, 0.5]
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result.x, expected, atol=1e-2, rtol=1e-2)

    def test_result_type(self):
        """Test that result is an OptimizeResult."""
        Q = torch.eye(2)
        c = torch.zeros(2)
        result = quadratic_program(Q, c)
        assert isinstance(result, OptimizeResult)
        assert result.x is not None
        assert result.converged is not None
        assert result.fun is not None

    def test_fun_value(self):
        """Test objective value at solution."""
        Q = torch.eye(2)
        c = torch.tensor([-1.0, -2.0])
        result = quadratic_program(Q, c)
        # f(x*) = 0.5 * [1,2] @ I @ [1,2] + [-1,-2] @ [1,2] = 2.5 - 5 = -2.5
        torch.testing.assert_close(
            result.fun,
            torch.tensor(-2.5),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_positive_definite(self):
        """Non-identity PSD matrix."""
        Q = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        c = torch.tensor([-1.0, -1.0])
        result = quadratic_program(Q, c)
        # x* = Q^{-1} (-c)
        expected = torch.linalg.solve(Q, -c)
        torch.testing.assert_close(result.x, expected, atol=1e-4, rtol=1e-4)


class TestQuadraticProgramAutograd:
    def test_implicit_diff_cost(self):
        """Gradient w.r.t. cost vector c."""
        Q = torch.eye(2)
        c = torch.tensor([-1.0, -2.0], requires_grad=True)
        result = quadratic_program(Q, c)
        result.x.sum().backward()
        assert c.grad is not None
        # dx*/dc = -Q^{-1} = -I, so d(sum(x*))/dc = [-1, -1]
        torch.testing.assert_close(
            c.grad,
            torch.tensor([-1.0, -1.0]),
            atol=1e-2,
            rtol=1e-2,
        )


class TestQuadraticProgramDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        Q = torch.eye(2, dtype=dtype)
        c = torch.zeros(2, dtype=dtype)
        result = quadratic_program(Q, c)
        assert result.x.dtype == dtype
