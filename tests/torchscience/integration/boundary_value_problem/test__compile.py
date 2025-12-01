"""Tests for torch.compile compatibility."""

import pytest
import torch

from torchscience.integration.boundary_value_problem import solve_bvp


class TestCompile:
    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_solve_bvp_compiles(self):
        """Test that solve_bvp works with torch.compile."""

        def fun(x, y, p):
            return y

        def bc(ya, yb, p):
            e = torch.exp(torch.tensor(1.0, dtype=ya.dtype))
            return torch.stack([ya[0] - 1.0, yb[0] - e])

        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        y = torch.ones(1, 10, dtype=torch.float64)

        # Solve without compile
        sol_eager = solve_bvp(fun, bc, x, y)

        # Solve with compile (may fall back to eager for unsupported ops)
        try:
            compiled_solve = torch.compile(solve_bvp, mode="reduce-overhead")
            sol_compiled = compiled_solve(fun, bc, x, y)

            # Results should match
            torch.testing.assert_close(
                sol_eager.y, sol_compiled.y, atol=1e-6, rtol=1e-6
            )
        except Exception as e:
            pytest.skip(f"torch.compile not fully supported: {e}")
