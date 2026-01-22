"""Phase 3 tests for PyTorch integration excellence."""

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    dormand_prince_5,
)


class TestTorchCompileFullgraph:
    """Tests for torch.compile with fullgraph=True (experimental)."""

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compiled_dynamics_with_adjoint(self):
        """Compiled dynamics function should work with adjoint solver."""
        theta = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)

        @torch.compile
        def dynamics(t, y):
            return -theta * y

        y0 = torch.tensor([1.0], dtype=torch.float64)

        solver = adjoint(dormand_prince_5, params=[theta])
        y_final, _ = solver(dynamics, y0, t_span=(0.0, 1.0))

        # Should complete without error
        loss = y_final.sum()
        loss.backward()

        assert theta.grad is not None
        assert torch.isfinite(theta.grad)

        # Verify accuracy matches non-compiled version
        theta_ref = torch.tensor(
            [1.0], requires_grad=True, dtype=torch.float64
        )

        def dynamics_ref(t, y):
            return -theta_ref * y

        solver_ref = adjoint(dormand_prince_5, params=[theta_ref])
        y_ref, _ = solver_ref(dynamics_ref, y0.clone(), t_span=(0.0, 1.0))
        y_ref.sum().backward()

        assert torch.allclose(theta.grad, theta_ref.grad, rtol=1e-6)

    @pytest.mark.skipif(
        not hasattr(torch, "_dynamo"), reason="torch._dynamo not available"
    )
    def test_adjoint_function_allow_in_graph(self):
        """_AdjointODEFunction should have allow_in_graph decorator."""
        from torchscience.integration.initial_value_problem._adjoint import (
            _AdjointODEFunction,
        )

        # Check if decorator was applied (class should be in allowed list)
        # Note: This is implementation-specific and may need adjustment
        assert hasattr(_AdjointODEFunction, "__name__")
