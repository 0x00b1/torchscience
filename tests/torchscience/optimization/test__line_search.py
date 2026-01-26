import torch
import torch.testing

from torchscience.optimization._line_search import (
    _backtracking_line_search,
    _cubic_interpolate,
    _strong_wolfe_line_search,
)


class TestCubicInterpolate:
    def test_quadratic_minimum(self):
        """Cubic interpolation of a quadratic finds the exact minimum."""
        # f(x) = (x - 2)^2 => f'(x) = 2(x - 2)
        x1 = torch.tensor(0.0)
        f1 = torch.tensor(4.0)
        g1 = torch.tensor(-4.0)
        x2 = torch.tensor(4.0)
        f2 = torch.tensor(4.0)
        g2 = torch.tensor(4.0)

        result = _cubic_interpolate(x1, f1, g1, x2, f2, g2)
        torch.testing.assert_close(
            result, torch.tensor(2.0), atol=1e-5, rtol=1e-5
        )


class TestBacktrackingLineSearch:
    def test_quadratic(self):
        """Test on simple quadratic f(x) = x^2."""

        def f(x):
            return (x**2).sum()

        x = torch.tensor([2.0])
        direction = torch.tensor([-1.0])
        grad_dot_dir = torch.tensor(-4.0)  # 2*x * direction = 2*2*(-1)
        f_x = torch.tensor(4.0)

        alpha = _backtracking_line_search(f, x, direction, grad_dot_dir, f_x)
        # Alpha should give sufficient decrease
        x_new = x + alpha * direction
        f_new = f(x_new)
        assert f_new < f_x

    def test_armijo_condition(self):
        """Verify the Armijo condition holds."""
        c1 = 1e-4

        def f(x):
            return (x**2).sum()

        x = torch.tensor([5.0])
        direction = torch.tensor([-1.0])
        grad_dot_dir = torch.tensor(-10.0)
        f_x = torch.tensor(25.0)

        alpha = _backtracking_line_search(
            f,
            x,
            direction,
            grad_dot_dir,
            f_x,
            c1=c1,
        )
        f_new = f(x + alpha * direction)
        assert f_new <= f_x + c1 * alpha * grad_dot_dir

    def test_full_step_accepted(self):
        """Test that alpha=1 is accepted when it satisfies Armijo."""

        def f(x):
            return (x**2).sum()

        # At x=0.5, direction=-0.5 (Newton step to minimum)
        x = torch.tensor([0.5])
        direction = torch.tensor([-0.5])
        grad_dot_dir = torch.tensor(-0.5)  # grad = 1.0, dir = -0.5
        f_x = torch.tensor(0.25)

        alpha = _backtracking_line_search(f, x, direction, grad_dot_dir, f_x)
        # Full step should be accepted (leads to x=0, f=0)
        torch.testing.assert_close(
            alpha, torch.tensor(1.0), atol=1e-5, rtol=1e-5
        )

    def test_batched(self):
        """Test batched line search."""

        def f(x):
            return (x**2).sum(dim=-1)

        x = torch.tensor([[3.0], [5.0]])  # (2, 1)
        direction = torch.tensor([[-1.0], [-1.0]])  # (2, 1)
        grad_dot_dir = torch.tensor([-6.0, -10.0])  # (2,)
        f_x = torch.tensor([9.0, 25.0])  # (2,)

        alpha = _backtracking_line_search(f, x, direction, grad_dot_dir, f_x)
        assert alpha.shape == (2,)

        # Both should satisfy Armijo
        for i in range(2):
            x_new = x[i] + alpha[i] * direction[i]
            f_new = (x_new**2).sum()
            assert f_new < f_x[i]


class TestStrongWolfeLineSearch:
    def _make_f_and_grad(self, f_fn):
        """Helper to create f_and_grad callable."""

        def f_and_grad(x):
            x = x.detach().requires_grad_(True)
            val = f_fn(x)
            grad = torch.autograd.grad(val, x)[0]
            return val.detach(), grad.detach()

        return f_and_grad

    def test_quadratic(self):
        """Test on quadratic f(x) = ||x||^2."""

        def f_fn(x):
            return (x**2).sum()

        f_and_grad = self._make_f_and_grad(f_fn)
        x = torch.tensor([2.0])
        direction = torch.tensor([-1.0])
        f_x = torch.tensor(4.0)
        grad_x = torch.tensor([4.0])

        alpha = _strong_wolfe_line_search(
            f_and_grad,
            x,
            direction,
            f_x,
            grad_x,
        )
        assert alpha > 0
        # Should produce decrease
        x_new = x + alpha * direction
        f_new = (x_new**2).sum()
        assert f_new < f_x

    def test_wolfe_conditions(self):
        """Verify both strong Wolfe conditions hold."""
        c1 = 1e-4
        c2 = 0.9

        def f_fn(x):
            return (x**2).sum()

        f_and_grad = self._make_f_and_grad(f_fn)
        x = torch.tensor([3.0])
        direction = torch.tensor([-1.0])
        f_x, grad_x = f_and_grad(x)

        alpha = _strong_wolfe_line_search(
            f_and_grad,
            x,
            direction,
            f_x,
            grad_x,
            c1=c1,
            c2=c2,
        )

        dg0 = torch.dot(grad_x, direction)
        x_new = x + alpha * direction
        f_new, grad_new = f_and_grad(x_new)
        dg_new = torch.dot(grad_new, direction)

        # Sufficient decrease (Armijo)
        assert f_new <= f_x + c1 * alpha * dg0
        # Curvature condition (strong Wolfe)
        assert torch.abs(dg_new) <= c2 * torch.abs(dg0)

    def test_rosenbrock_direction(self):
        """Test on Rosenbrock function with steepest descent direction."""

        def f_fn(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

        f_and_grad = self._make_f_and_grad(f_fn)
        x = torch.tensor([-1.0, 1.0])
        f_x, grad_x = f_and_grad(x)
        direction = -grad_x  # Steepest descent

        alpha = _strong_wolfe_line_search(
            f_and_grad,
            x,
            direction,
            f_x,
            grad_x,
        )
        assert alpha > 0
        x_new = x + alpha * direction
        f_new = (1 - x_new[0]) ** 2 + 100 * (x_new[1] - x_new[0] ** 2) ** 2
        assert f_new < f_x
