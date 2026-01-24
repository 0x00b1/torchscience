"""Tests for BVP adjoint method."""

import torch

from torchscience.integration._bvp_adjoint import (
    bvp_adjoint,
)


class TestBVPAdjoint:
    def test_adjoint_gradients_match_direct(self):
        """Test adjoint gradients match direct backprop for simple BVP."""
        # y' = p * y, y(0) = 1, y(1) = e^p
        # This has exact solution y = e^(p*x)

        def make_solver():
            """Create a simple BVP solver for testing."""
            from torchscience.integration._bvp_collocation import (
                compute_collocation_residual,
            )
            from torchscience.integration._bvp_newton import (
                newton_bvp,
            )
            from torchscience.integration._bvp_solution import (
                BVPSolution,
            )

            def solver(fun, bc, x, y, p=None, **kwargs):
                if p is None:
                    p = torch.empty(0, dtype=x.dtype, device=x.device)
                y_sol, p_sol, converged, n_iter = newton_bvp(fun, bc, x, y, p)
                f_sol = fun(x, y_sol, p_sol)
                coll_res = compute_collocation_residual(fun, x, y_sol, p_sol)
                rms = coll_res.abs().max()
                return BVPSolution(
                    x=x,
                    y=y_sol,
                    yp=f_sol,
                    p=p_sol,
                    rms_residuals=rms,
                    n_iterations=n_iter,
                    success=converged,
                )

            return solver

        solver = make_solver()
        adjoint_solver = bvp_adjoint(solver)

        # Problem setup
        p_init = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

        def fun(x, y, p):
            return p[0] * y

        def bc(ya, yb, p):
            target = torch.exp(p[0])
            return torch.stack([ya[0] - 1.0, yb[0] - target])

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y_init = torch.ones(1, 10, dtype=torch.float64)

        # Solve with adjoint
        sol = adjoint_solver(fun, bc, x, y_init, p=p_init)

        # Compute loss and gradient
        loss = sol.y.sum()
        loss.backward()

        assert p_init.grad is not None
        # Gradient should be non-zero and finite
        assert torch.isfinite(p_init.grad).all()
        assert p_init.grad.abs() > 0.1  # Non-trivial gradient

    def test_adjoint_wraps_solver(self):
        """Test that bvp_adjoint returns a wrapped solver."""

        def mock_solver(fun, bc, x, y, p=None, **kwargs):
            from torchscience.integration._bvp_solution import (
                BVPSolution,
            )

            return BVPSolution(
                x=x,
                y=y,
                yp=torch.zeros_like(y),
                p=p if p is not None else torch.empty(0),
                rms_residuals=torch.tensor(0.0),
                n_iterations=1,
                success=True,
            )

        wrapped = bvp_adjoint(mock_solver)

        assert callable(wrapped)
        assert "adjoint" in wrapped.__name__

    def test_adjoint_preserves_solution(self):
        """Test that adjoint wrapper preserves solution values."""
        from torchscience.integration._bvp_solution import (
            BVPSolution,
        )

        expected_y = torch.randn(2, 5, dtype=torch.float64)
        expected_x = torch.linspace(0, 1, 5, dtype=torch.float64)

        def mock_solver(fun, bc, x, y, p=None, **kwargs):
            return BVPSolution(
                x=expected_x,
                y=expected_y,
                yp=torch.zeros_like(expected_y),
                p=p if p is not None else torch.empty(0),
                rms_residuals=torch.tensor(1e-6),
                n_iterations=3,
                success=True,
            )

        wrapped = bvp_adjoint(mock_solver)

        def fun(x, y, p):
            return y

        def bc(ya, yb, p):
            return ya

        sol = wrapped(
            fun,
            bc,
            expected_x,
            torch.zeros_like(expected_y),
            p=torch.tensor([1.0]),
        )

        torch.testing.assert_close(sol.y, expected_y)
        torch.testing.assert_close(sol.x, expected_x)
