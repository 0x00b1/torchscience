# tests/torchscience/integration/initial_value_problem/test__newton.py
import torch

from torchscience.ordinary_differential_equation._newton import newton_solve


class TestNewtonSolve:
    def test_simple_root(self):
        """Find root of f(x) = x^2 - 2 => x = sqrt(2)"""

        def f(x):
            return x**2 - 2

        # Use float64 for tight tolerance convergence
        x0 = torch.tensor([1.0], dtype=torch.float64)
        x_root, converged = newton_solve(f, x0, tol=1e-10, max_iter=50)

        expected = torch.sqrt(torch.tensor([2.0], dtype=torch.float64))
        assert converged
        assert torch.allclose(x_root, expected, atol=1e-8)

    def test_multidimensional(self):
        """Find root of [x^2 + y - 1, x + y^2 - 1] => x = y = 0.6180..."""

        def f(z):
            x, y = z[0], z[1]
            return torch.stack([x**2 + y - 1, x + y**2 - 1])

        # Start away from singular point (0.5, 0.5) where Jacobian is singular
        z0 = torch.tensor([0.6, 0.7], dtype=torch.float64)
        z_root, converged = newton_solve(f, z0, tol=1e-10, max_iter=50)

        # Both roots are golden ratio related
        assert converged
        assert torch.allclose(
            f(z_root), torch.zeros(2, dtype=torch.float64), atol=1e-8
        )

    def test_convergence_failure(self):
        """Should not converge with too few iterations"""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([100.0])  # Far from root
        _, converged = newton_solve(f, x0, tol=1e-10, max_iter=2)

        assert not converged

    def test_batched(self):
        """Solve multiple systems in parallel"""
        targets = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float64)

        def f(x):
            return x**2 - targets

        x0 = torch.tensor([[1.0], [1.5], [1.8]], dtype=torch.float64)
        x_root, converged = newton_solve(f, x0, tol=1e-8, max_iter=50)

        expected = torch.sqrt(targets)
        assert converged
        assert torch.allclose(x_root, expected, atol=1e-6)

    def test_differentiable(self):
        """Gradients should flow through the solution"""
        a = torch.tensor([2.0], requires_grad=True)

        def f(x):
            return x**2 - a

        x0 = torch.tensor([1.0])
        x_root, _ = newton_solve(f, x0, tol=1e-10, max_iter=50)

        # x_root = sqrt(a), so d(x_root)/da = 1/(2*sqrt(a))
        loss = x_root.sum()
        loss.backward()

        expected_grad = 1 / (2 * torch.sqrt(a))
        assert a.grad is not None
        assert torch.allclose(a.grad, expected_grad, rtol=1e-4)
