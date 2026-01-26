import torch

from torchscience.optimization._result import OptimizeResult


class TestOptimizeResult:
    def test_field_access(self):
        """Test accessing result fields by name."""
        x = torch.tensor([1.0, 2.0])
        converged = torch.tensor(True)
        num_iterations = torch.tensor(10, dtype=torch.int64)

        result = OptimizeResult(
            x=x,
            converged=converged,
            num_iterations=num_iterations,
        )

        torch.testing.assert_close(result.x, x)
        assert result.converged.item() is True
        assert result.num_iterations.item() == 10
        assert result.fun is None

    def test_unpacking(self):
        """Test tuple unpacking."""
        x = torch.tensor([1.0])
        converged = torch.tensor(True)
        num_iterations = torch.tensor(5, dtype=torch.int64)
        fun = torch.tensor(0.0)

        result = OptimizeResult(
            x=x,
            converged=converged,
            num_iterations=num_iterations,
            fun=fun,
        )

        x_out, conv_out, nit_out, fun_out = result
        torch.testing.assert_close(x_out, x)
        assert conv_out.item() is True
        assert nit_out.item() == 5
        torch.testing.assert_close(fun_out, fun)

    def test_optional_fun(self):
        """Test that fun defaults to None."""
        result = OptimizeResult(
            x=torch.zeros(2),
            converged=torch.tensor(False),
            num_iterations=torch.tensor(0, dtype=torch.int64),
        )
        assert result.fun is None

    def test_fun_provided(self):
        """Test that fun can be provided."""
        fun_val = torch.tensor(3.14)
        result = OptimizeResult(
            x=torch.zeros(2),
            converged=torch.tensor(True),
            num_iterations=torch.tensor(42, dtype=torch.int64),
            fun=fun_val,
        )
        torch.testing.assert_close(result.fun, fun_val)

    def test_batched_converged(self):
        """Test batch-shaped converged tensor."""
        converged = torch.tensor([True, False, True])
        result = OptimizeResult(
            x=torch.zeros(3, 2),
            converged=converged,
            num_iterations=torch.tensor(10, dtype=torch.int64),
        )
        assert result.converged.shape == (3,)
        assert result.converged[0].item() is True
        assert result.converged[1].item() is False
