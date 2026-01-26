import pytest
import torch
import torch.testing

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.linear._linear_program import linear_program


class TestLinearProgram:
    def test_simple_2d(self):
        """min -x1 - x2 s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3, x >= 0."""
        c = torch.tensor([-1.0, -1.0])
        A_ub = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b_ub = torch.tensor([4.0, 3.0, 3.0])
        result = linear_program(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(torch.zeros(2), None),
        )
        # Optimal: x = [3, 1] or [1, 3]; objective = -4
        assert result.fun.item() < -3.9

    def test_equality_constraint(self):
        """min c^T x s.t. A_eq x = b_eq, x >= 0."""
        c = torch.tensor([-1.0, -2.0])
        A_eq = torch.tensor([[1.0, 1.0]])
        b_eq = torch.tensor([1.0])
        result = linear_program(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(torch.zeros(2), None),
        )
        # Optimal: x = [0, 1], objective = -2
        torch.testing.assert_close(
            result.x,
            torch.tensor([0.0, 1.0]),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_result_type(self):
        """Test that result is an OptimizeResult."""
        c = torch.tensor([-1.0])
        A_ub = torch.tensor([[1.0]])
        b_ub = torch.tensor([5.0])
        result = linear_program(
            c, A_ub=A_ub, b_ub=b_ub, bounds=(torch.zeros(1), None)
        )
        assert isinstance(result, OptimizeResult)

    def test_convergence_flag(self):
        """Test convergence flag."""
        c = torch.tensor([-1.0, -2.0])
        A_eq = torch.tensor([[1.0, 1.0]])
        b_eq = torch.tensor([1.0])
        result = linear_program(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(torch.zeros(2), None),
        )
        assert result.converged.item() is True


class TestLinearProgramDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""
        c = torch.tensor([-1.0], dtype=dtype)
        A_ub = torch.tensor([[1.0]], dtype=dtype)
        b_ub = torch.tensor([5.0], dtype=dtype)
        result = linear_program(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=(torch.zeros(1, dtype=dtype), None),
        )
        assert result.x.dtype == dtype
