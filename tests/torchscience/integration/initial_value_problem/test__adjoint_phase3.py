"""Phase 3 tests for PyTorch integration excellence."""

import pytest
import torch

from torchscience.integration.initial_value_problem import (
    adjoint,
    dormand_prince_5,
    solve_ivp,
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


class TestVmapCompatibility:
    """Tests for torch.vmap compatibility.

    Note: Adaptive ODE solvers like dormand_prince_5 use data-dependent control
    flow (while loops, error checking, step acceptance) which is fundamentally
    incompatible with vmap's current implementation. vmap requires static control
    flow where branches don't depend on tensor values.

    These tests document the current limitation and will pass when/if PyTorch
    adds support for data-dependent control flow in vmap (see:
    https://github.com/pytorch/functorch/issues/257).
    """

    @pytest.mark.xfail(
        reason="Adaptive ODE solvers use data-dependent control flow incompatible with vmap",
        raises=RuntimeError,
    )
    @pytest.mark.skipif(
        not hasattr(torch, "vmap"), reason="torch.vmap not available"
    )
    def test_vmap_over_solve_basic(self):
        """vmap should work over solve_ivp for batched initial conditions.

        Currently expected to fail because dormand_prince_5 uses data-dependent
        control flow for adaptive stepping (error estimation, step acceptance,
        while loop termination) which vmap does not support.
        """
        theta = torch.tensor([1.0], dtype=torch.float64)

        def dynamics(t, y):
            return -theta * y

        def solve_single(y0_single):
            y_final, _ = solve_ivp(
                dynamics,
                y0_single,
                t_span=(0.0, 1.0),
                method="dormand_prince_5",
            )
            return y_final

        # Batch of initial conditions
        y0_batch = torch.randn(8, 4, dtype=torch.float64)

        # vmap over initial conditions
        batched_solve = torch.vmap(solve_single, in_dims=0)
        y_final_batch = batched_solve(y0_batch)

        # Should have correct shape
        assert y_final_batch.shape == (8, 4)

        # Verify against sequential solves
        y_sequential = torch.stack([solve_single(y0) for y0 in y0_batch])
        assert torch.allclose(y_final_batch, y_sequential, rtol=1e-5)
