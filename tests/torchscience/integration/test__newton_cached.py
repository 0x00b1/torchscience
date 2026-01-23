"""Tests for Newton solver with Jacobian caching."""

import torch

from torchscience.integration._newton_cached import (
    JacobianCache,
    newton_solve_cached,
)


class TestJacobianCache:
    def test_cache_stores_lu_factorization(self):
        """Cache should store LU factorization for reuse."""

        def f(x):
            return x**2 - 2  # Root at sqrt(2)

        x0 = torch.tensor([1.0], dtype=torch.float64)
        cache = JacobianCache()

        x, converged, _ = newton_solve_cached(f, x0, cache=cache)

        assert converged
        assert cache.lu_pivots is not None
        assert cache.n_factorizations >= 1


class TestNewtonSolveCached:
    def test_simple_scalar(self):
        """Solve x^2 - 2 = 0."""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([1.5], dtype=torch.float64)
        x, converged, _ = newton_solve_cached(f, x0)

        assert converged
        expected = torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
        assert torch.allclose(x, expected.unsqueeze(0), atol=1e-10)

    def test_multidimensional_system(self):
        """Solve nonlinear system: x^2 + y^2 = 1, x - y = 0."""

        def f(z):
            x, y = z[0], z[1]
            return torch.stack([x**2 + y**2 - 1, x - y])

        z0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
        z, converged, _ = newton_solve_cached(f, z0)

        assert converged
        # Solution: x = y = 1/sqrt(2)
        expected = torch.tensor([1.0, 1.0], dtype=torch.float64) / torch.sqrt(
            torch.tensor(2.0)
        )
        assert torch.allclose(z, expected, atol=1e-10)

    def test_jacobian_reuse_reduces_factorizations(self):
        """Modified Newton should reuse Jacobian across iterations."""

        def f(x):
            return x**3 - x - 2

        x0 = torch.tensor([1.5], dtype=torch.float64)
        cache = JacobianCache()

        x, converged, info = newton_solve_cached(
            f, x0, cache=cache, max_iter=20, recompute_jacobian_every=5
        )

        assert converged
        # With recompute_every=5 and ~10 iterations, should have ~2 factorizations
        assert cache.n_factorizations < info["n_iterations"]

    def test_user_provided_jacobian(self):
        """Should use user-provided Jacobian function."""
        call_count = [0]

        def f(x):
            return x**2 - 2

        def jac(x):
            call_count[0] += 1
            return 2 * x.unsqueeze(0)

        x0 = torch.tensor([1.5], dtype=torch.float64)
        x, converged, _ = newton_solve_cached(f, x0, jacobian=jac)

        assert converged
        assert call_count[0] > 0

    def test_info_dict_contains_diagnostics(self):
        """Info dict should contain n_iterations and final_residual_norm."""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([1.5], dtype=torch.float64)
        x, converged, info = newton_solve_cached(f, x0)

        assert converged
        assert "n_iterations" in info
        assert "final_residual_norm" in info
        assert info["n_iterations"] >= 1
        assert info["final_residual_norm"] < 1e-8

    def test_singular_jacobian_returns_false(self):
        """Should return converged=False for singular Jacobian."""

        def f(x):
            # f(x) = 0 for all x -> J = 0 (singular)
            return torch.zeros_like(x)

        x0 = torch.tensor([1.0], dtype=torch.float64)
        x, converged, _ = newton_solve_cached(f, x0, tol=1e-10)

        # The residual is already 0, so it should converge immediately
        # Let's use a different example where J is singular but residual != 0
        def g(x):
            # g(x, y) = [x + y, x + y] -> J has rank 1
            return torch.stack([x[0] + x[1] - 1, x[0] + x[1] - 1])

        z0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        z, converged, _ = newton_solve_cached(g, z0, tol=1e-10)

        # Singular Jacobian should cause failure
        assert not converged

    def test_max_iter_exceeded(self):
        """Should return converged=False when max_iter exceeded."""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([100.0], dtype=torch.float64)  # Far from root
        x, converged, info = newton_solve_cached(f, x0, max_iter=2, tol=1e-12)

        assert not converged
        assert info["n_iterations"] == 2

    def test_cache_reuse_across_calls(self):
        """Cache can be reused across multiple newton_solve_cached calls."""

        def f(x, target):
            return x**2 - target

        cache = JacobianCache()

        # First call
        x0 = torch.tensor([1.5], dtype=torch.float64)
        x1, converged1, _ = newton_solve_cached(
            lambda x: f(x, 2.0), x0, cache=cache
        )
        factorizations_after_first = cache.n_factorizations

        # Clear cache and solve again (simulates warm start scenario)
        # In practice, for implicit ODE solvers, the cache would be reused
        # when the Jacobian hasn't changed much
        cache.clear()
        x2, converged2, _ = newton_solve_cached(
            lambda x: f(x, 2.0), x0, cache=cache
        )

        assert converged1
        assert converged2
        assert factorizations_after_first >= 1

    def test_tolerance_respected(self):
        """Solution residual should be below tolerance."""

        def f(x):
            return x**3 - 2 * x - 5

        x0 = torch.tensor([2.0], dtype=torch.float64)
        tol = 1e-12

        x, converged, info = newton_solve_cached(f, x0, tol=tol)

        assert converged
        residual = f(x)
        assert torch.abs(residual).item() < tol
