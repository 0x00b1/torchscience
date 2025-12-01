import pytest
import torch
import torch.testing

import torchscience.optimization.constrained


class TestAugmentedLagrangian:
    def test_unconstrained_quadratic(self):
        """Without constraints, should minimize f(x) = ||x - target||²."""
        target = torch.tensor([1.0, 2.0])

        def objective(x):
            return torch.sum((x - target) ** 2)

        x0 = torch.zeros(2)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0
        )
        torch.testing.assert_close(result, target, atol=1e-4, rtol=1e-4)

    def test_equality_constraint(self):
        """Minimize x² + y² subject to x + y = 1."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0  # h(x) = x + y - 1 = 0

        x0 = torch.zeros(2)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0, eq_constraints=eq_constraints
        )
        # Optimal: x = y = 0.5
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_inequality_constraint(self):
        """Minimize -x subject to x <= 2."""

        def objective(x):
            return -x.sum()

        def ineq_constraints(x):
            return x - 2.0  # g(x) = x - 2 <= 0

        x0 = torch.zeros(1)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0, ineq_constraints=ineq_constraints
        )
        # Optimal: x = 2 (at the constraint boundary)
        expected = torch.tensor([2.0])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_mixed_constraints(self):
        """Minimize x² + y² subject to x + y = 1, x >= 0.3."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        def ineq_constraints(x):
            return 0.3 - x[0]  # -x + 0.3 <= 0, i.e., x >= 0.3

        x0 = torch.tensor([0.5, 0.5])
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective,
            x0,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        # Without inequality: x=y=0.5. With x>=0.3, constraint is inactive.
        expected = torch.tensor([0.5, 0.5])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    def test_active_inequality(self):
        """Minimize x² + y² subject to x + y = 1, x >= 0.6."""

        def objective(x):
            return torch.sum(x**2)

        def eq_constraints(x):
            return x.sum() - 1.0

        def ineq_constraints(x):
            return 0.6 - x[0]  # x >= 0.6

        x0 = torch.tensor([0.7, 0.3])
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective,
            x0,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )
        # Constrained optimum: x=0.6, y=0.4
        expected = torch.tensor([0.6, 0.4])
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.xfail(
        reason="Implicit differentiation through KKT not yet fully implemented"
    )
    def test_implicit_differentiation(self):
        """Test gradient through constrained optimizer."""
        target = torch.tensor([1.0], requires_grad=True)

        def objective(x):
            return (x - target) ** 2

        def eq_constraints(x):
            return x - 0.5  # x = 0.5

        x0 = torch.zeros(1)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0, eq_constraints=eq_constraints
        )
        # Result is always 0.5 regardless of target, so gradient should be 0
        loss = result.sum()
        loss.backward()
        torch.testing.assert_close(
            target.grad, torch.tensor([0.0]), atol=1e-4, rtol=1e-4
        )

    @pytest.mark.xfail(
        reason="Rosenbrock is nonconvex, gradient descent inner loop struggles"
    )
    def test_rosenbrock_constrained(self):
        """Minimize Rosenbrock subject to x² + y² <= 2.

        Note: Rosenbrock has a narrow curved valley making it challenging
        for first-order methods. A proper implementation would use a
        quasi-Newton inner solver (e.g., L-BFGS).
        """

        def objective(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        def ineq_constraints(x):
            return torch.sum(x**2) - 2.0  # ||x||² <= 2

        # Start closer to optimum for better convergence
        x0 = torch.tensor([0.5, 0.5])
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective,
            x0,
            ineq_constraints=ineq_constraints,
            maxiter=100,
            inner_maxiter=200,
        )
        # Unconstrained optimum is (1, 1) with ||x||² = 2, exactly on boundary
        expected = torch.tensor([1.0, 1.0])
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)


class TestAugmentedLagrangianDtypes:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that output dtype matches input."""

        def objective(x):
            return torch.sum(x**2)

        x0 = torch.ones(2, dtype=dtype)
        result = torchscience.optimization.constrained.augmented_lagrangian(
            objective, x0
        )
        assert result.dtype == dtype
