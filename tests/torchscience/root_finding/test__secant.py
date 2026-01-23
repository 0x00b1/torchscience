# tests/torchscience/root_finding/test__secant.py
import math

import pytest
import torch

from torchscience.root_finding import secant

# Check if scipy is available for comparison tests
try:
    from scipy.optimize import newton as scipy_newton

    scipy_available = True
except ImportError:
    scipy_available = False


class TestSecant:
    """Tests for Secant root-finding method."""

    def test_simple_quadratic(self):
        """Find sqrt(2) by solving x^2 - 2 = 0."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])

        root, converged = secant(f, x0)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_auto_x1(self):
        """Works when x1 not provided (auto-generated)."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])

        # Should work without x1
        root, converged = secant(f, x0)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_explicit_x1(self):
        """Works when x1 is explicitly provided."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.0])
        x1 = torch.tensor([2.0])

        root, converged = secant(f, x0, x1)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_batched(self):
        """Find multiple roots in parallel."""
        c = torch.tensor([2.0, 3.0, 4.0, 5.0])
        f = lambda x: x**2 - c
        x0 = torch.full((4,), 1.5)

        roots, converged = secant(f, x0)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_superlinear_convergence(self):
        """Secant method converges reasonably fast (superlinear, ~1.618 order)."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float64)

        # With maxiter=10, secant should converge for a simple function
        root, converged = secant(f, x0, maxiter=10)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_preserves_shape(self):
        """2D input preserves shape in output."""
        f = lambda x: x**2 - 2
        x0 = torch.full((2, 3), 1.5)

        root, converged = secant(f, x0)

        assert root.shape == (2, 3)
        assert converged.shape == (2, 3)
        expected = torch.full((2, 3), math.sqrt(2))
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_returns_tuple(self):
        """secant returns (root, converged) tuple."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])

        result = secant(f, x0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        root, converged = result
        assert root.shape == x0.shape
        assert converged.shape == x0.shape
        assert converged.dtype == torch.bool

    def test_float64(self):
        """Works with float64 and achieves higher precision."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = secant(f, x0)

        assert root.dtype == torch.float64
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_float32(self):
        """Works correctly with float32."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float32)

        root, converged = secant(f, x0)

        assert root.dtype == torch.float32
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        assert converged.all()

    def test_trigonometric(self):
        """Find root of sin(x) = 0 near pi."""
        f = lambda x: torch.sin(x)
        x0 = torch.tensor([3.0])

        root, converged = secant(f, x0)

        torch.testing.assert_close(
            root, torch.tensor([math.pi]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_cubic(self):
        """Find root of cubic equation."""
        f = lambda x: x**3 - x - 1
        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = secant(f, x0)

        # Verify the root satisfies f(root) = 0
        residual = f(root)
        assert torch.abs(residual) < 1e-10
        assert converged.all()

    def test_transcendental(self):
        """Find root of exp(x) - 2 = 0."""
        f = lambda x: torch.exp(x) - 2
        x0 = torch.tensor([1.0], dtype=torch.float64)

        root, converged = secant(f, x0)

        expected = math.log(2)
        torch.testing.assert_close(
            root,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_empty_input(self):
        """Handle empty input gracefully."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([])

        root, converged = secant(f, x0)

        assert root.shape == (0,)
        assert converged.shape == (0,)

    def test_zero_x0(self):
        """Works when x0 is zero (auto x1 still works)."""
        f = lambda x: x**2 - 4
        x0 = torch.tensor([0.0])

        root, converged = secant(f, x0)

        # Should find root = 2 (positive root near perturbed x1)
        assert torch.isfinite(root).all()
        # The root could be +2 or we might not converge perfectly from 0
        # but the result should be finite

    def test_zero_denominator_safeguard(self):
        """Safeguards against zero denominator by using eps * sign(denom).

        When f(x_n) - f(x_{n-1}) is very small, the safeguard should prevent
        NaN or Inf from appearing in the result.
        """
        # Function where we might get equal f values
        f = lambda x: x**3
        x0 = torch.tensor([0.1])

        root, converged = secant(f, x0, maxiter=100)

        # The key thing is that we don't get NaN or Inf
        assert torch.isfinite(root).all(), f"Expected finite root, got {root}"

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy(self):
        """Results match scipy.optimize.newton (secant mode)."""
        f_torch = lambda x: x**2 - 2
        f_scipy = lambda x: x**2 - 2

        # scipy.optimize.newton uses secant when fprime is not provided
        scipy_root = scipy_newton(f_scipy, 1.5)
        our_root, converged = secant(
            f_torch, torch.tensor([1.5], dtype=torch.float64)
        )

        torch.testing.assert_close(
            our_root,
            torch.tensor([scipy_root], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy_transcendental(self):
        """Results match scipy.optimize.newton for transcendental function."""
        f_torch = lambda x: torch.exp(x) - 2
        f_scipy = lambda x: math.exp(x) - 2

        scipy_root = scipy_newton(f_scipy, 1.0)
        our_root, converged = secant(
            f_torch, torch.tensor([1.0], dtype=torch.float64)
        )

        torch.testing.assert_close(
            our_root,
            torch.tensor([scipy_root], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()


class TestSecantAutograd:
    """Tests for autograd support via implicit differentiation."""

    def test_implicit_diff(self):
        """Test gradient w.r.t. function parameter via implicit differentiation.

        For f(x) = x^2 - theta, root x* = sqrt(theta).
        By implicit function theorem: dx*/dtheta = -[df/dx]^{-1} * df/dtheta
        df/dx = 2x = 2*sqrt(theta), df/dtheta = -1
        dx*/dtheta = -1/(2*sqrt(theta)) * (-1) = 1/(2*sqrt(theta))
        """
        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        def f(x):
            return x**2 - theta

        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = secant(f, x0)
        root.sum().backward()

        # Expected: 1/(2*sqrt(2)) = 0.3536
        expected = 1.0 / (2.0 * math.sqrt(2.0))
        torch.testing.assert_close(
            theta.grad,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        assert converged.all()

    def test_implicit_diff_batched(self):
        """Test gradient with batched inputs."""
        theta = torch.tensor(
            [2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def f(x):
            return x**2 - theta

        x0 = torch.full((3,), 1.5, dtype=torch.float64)

        roots, converged = secant(f, x0)
        loss = roots.sum()
        loss.backward()

        # Expected: 1/(2*sqrt(theta_i)) for each element
        expected = 1.0 / (2.0 * torch.sqrt(theta.detach()))
        torch.testing.assert_close(theta.grad, expected, rtol=1e-4, atol=1e-6)
        assert converged.all()

    def test_implicit_diff_linear_param(self):
        """Test gradient with linear parameter dependence.

        For f(x) = x - theta, root x* = theta.
        df/dx = 1, df/dtheta = -1
        dx*/dtheta = -1/1 * (-1) = 1
        """
        theta = torch.tensor([5.0], dtype=torch.float64, requires_grad=True)

        def f(x):
            return x - theta

        x0 = torch.tensor([4.0], dtype=torch.float64)

        root, converged = secant(f, x0)
        root.sum().backward()

        # Expected: dx*/dtheta = 1
        torch.testing.assert_close(
            theta.grad,
            torch.tensor([1.0], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        assert converged.all()

    def test_implicit_diff_no_param_grad(self):
        """Test that no error occurs when function has no differentiable parameters."""
        constant = 2.0  # Not a tensor with requires_grad

        def f(x):
            return x**2 - constant

        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = secant(f, x0)

        # Should just return the root without error
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        assert converged.all()

    def test_implicit_diff_numerical_verification(self):
        """Verify gradient numerically (finite differences)."""
        theta_val = 2.0
        eps = 1e-5

        def f(x, t):
            return x**2 - t

        x0 = torch.tensor([1.5], dtype=torch.float64)

        # Compute root at theta and theta + eps
        theta = torch.tensor(
            [theta_val], dtype=torch.float64, requires_grad=True
        )

        def f1(x):
            return f(x, theta)

        root1, _ = secant(f1, x0)

        theta_plus = torch.tensor([theta_val + eps], dtype=torch.float64)

        def f2(x):
            return f(x, theta_plus)

        root2, _ = secant(f2, x0)

        # Numerical gradient
        numerical_grad = (root2 - root1) / eps

        # Analytical gradient via backward
        root1.sum().backward()
        analytical_grad = theta.grad

        torch.testing.assert_close(
            analytical_grad, numerical_grad, rtol=1e-3, atol=1e-6
        )

    def test_gradient_with_loss_function(self):
        """Test gradient computation through a loss function."""
        theta = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([1.5], dtype=torch.float64)

        def f(x):
            return x**2 - theta

        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = secant(f, x0)
        loss = (root - target) ** 2  # MSE loss

        loss.backward()

        # root = sqrt(theta) = 2.0
        # d(loss)/d(root) = 2 * (root - target) = 2 * (2.0 - 1.5) = 1.0
        # d(root)/d(theta) = 1/(2*sqrt(theta)) = 1/4 = 0.25
        # d(loss)/d(theta) = 1.0 * 0.25 = 0.25
        expected = torch.tensor([0.25], dtype=torch.float64)
        torch.testing.assert_close(theta.grad, expected, rtol=1e-4, atol=1e-6)
        assert converged.all()

    def test_gradgradcheck(self):
        """Test second-order gradients via gradgradcheck.

        For f(x) = x^2 - theta, root x* = sqrt(theta).
        First derivative: dx*/dtheta = 1/(2*sqrt(theta))
        Second derivative: d^2x*/dtheta^2 = -1/(4*theta^(3/2))
        """

        def func(theta):
            f = lambda x: x**2 - theta
            x0 = torch.tensor([1.5], dtype=torch.float64)
            root, _ = secant(f, x0)
            return root.sum()

        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(func, theta)

    def test_gradgradcheck_cubic(self):
        """Test second-order gradients with a cubic function.

        For f(x) = x^3 - theta, root x* = theta^(1/3).
        First derivative: dx*/dtheta = 1/(3*theta^(2/3))
        Second derivative: d^2x*/dtheta^2 = -2/(9*theta^(5/3))
        """

        def func(theta):
            f = lambda x: x**3 - theta
            x0 = torch.tensor([1.5], dtype=torch.float64)
            root, _ = secant(f, x0)
            return root.sum()

        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(func, theta)


class TestSecantVmap:
    """Tests for vmap compatibility.

    Note: vmap is currently NOT compatible with secant due to:
    1. Data-dependent control flow (if torch.all(converged)) not supported

    These tests document the expected incompatibility. Use explicit batching
    (e.g., secant(f, batched_x0)) instead of vmap for vectorized computation.
    """

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_basic(self):
        """vmap works with secant for vectorized parameter sweeps."""
        from torch.func import vmap

        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        def solve_one(ci):
            f = lambda x: x**2 - ci
            x0 = torch.tensor([1.5], dtype=torch.float64)
            root, converged = secant(f, x0)
            return root

        # Use vmap to vectorize over the first dimension
        roots = vmap(solve_one)(c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(
            roots.squeeze(-1), expected, rtol=1e-6, atol=1e-6
        )

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_different_starting_points(self):
        """vmap works with different starting points for the same problem."""
        from torch.func import vmap

        def solve_sqrt2(x0_scalar):
            x0 = x0_scalar.unsqueeze(0)
            f = lambda x: x**2 - 2.0
            root, converged = secant(f, x0)
            return root

        x0_vals = torch.tensor([1.0, 1.5, 2.0], dtype=torch.float64)
        roots = vmap(solve_sqrt2)(x0_vals)

        expected = torch.full((3, 1), math.sqrt(2), dtype=torch.float64)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_with_grad(self):
        """vmap + grad works together for parameter gradients."""
        from torch.func import grad, vmap

        def solve_and_return(ci):
            f = lambda x: x**2 - ci
            x0 = torch.tensor([1.5], dtype=torch.float64)
            root, _ = secant(f, x0)
            return root.sum()

        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        # grad of vmapped function
        grads = vmap(grad(solve_and_return))(c)

        # d(sqrt(c))/dc = 1/(2*sqrt(c))
        expected = 1.0 / (2.0 * torch.sqrt(c))
        torch.testing.assert_close(grads, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_explicit_x1(self):
        """vmap works with explicit x1."""
        from torch.func import vmap

        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        def solve_one(ci):
            f = lambda x: x**2 - ci
            x0 = torch.tensor([1.0], dtype=torch.float64)
            x1 = torch.tensor([2.0], dtype=torch.float64)
            root, converged = secant(f, x0, x1)
            return root

        roots = vmap(solve_one)(c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(
            roots.squeeze(-1), expected, rtol=1e-6, atol=1e-6
        )

    def test_explicit_batching_alternative(self):
        """Demonstrates explicit batching as alternative to vmap.

        Instead of using vmap, pass batched inputs directly to secant.
        """
        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        f = lambda x: x**2 - c
        x0 = torch.full((3,), 1.5, dtype=torch.float64)

        roots, converged = secant(f, x0)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()


class TestSecantCompile:
    """Tests for torch.compile compatibility.

    Note: torch.compile may have issues with data-dependent loops used in
    iterative root-finding algorithms. These tests verify compatibility
    and document any limitations.
    """

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_eager_backend(self):
        """torch.compile with eager backend works with secant."""
        c = torch.tensor([2.0, 3.0, 4.0])

        def solve(x0, c_val):
            f = lambda x: x**2 - c_val
            root, converged = secant(f, x0)
            return root

        # Compile with eager backend (more forgiving)
        solve_compiled = torch.compile(solve, backend="eager")

        x0 = torch.tensor([1.5, 1.5, 1.5])
        roots = solve_compiled(x0, c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(
        reason="torch.compile with inductor may not support data-dependent loops"
    )
    def test_compile_inductor_backend(self):
        """torch.compile with inductor backend works with secant."""
        c = torch.tensor([2.0, 3.0, 4.0])

        def solve(x0, c_val):
            f = lambda x: x**2 - c_val
            root, converged = secant(f, x0)
            return root

        # Compile with default inductor backend
        solve_compiled = torch.compile(solve)

        x0 = torch.tensor([1.5, 1.5, 1.5])
        roots = solve_compiled(x0, c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_with_grad_eager(self):
        """Compiled method with eager backend supports autograd."""
        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        @torch.compile(backend="eager")
        def solve_and_sum(t):
            f = lambda x: x**2 - t
            x0 = torch.tensor([1.5], dtype=torch.float64)
            root, _ = secant(f, x0)
            return root.sum()

        result = solve_and_sum(theta)
        result.backward()

        # d(sqrt(theta))/dtheta = 1/(2*sqrt(theta))
        expected_grad = 1.0 / (2.0 * torch.sqrt(theta.detach()))
        torch.testing.assert_close(
            theta.grad, expected_grad, rtol=1e-4, atol=1e-6
        )

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_explicit_x1(self):
        """torch.compile works with explicit x1 parameter."""
        c = torch.tensor([2.0, 3.0, 4.0])

        def solve(x0, x1, c_val):
            f = lambda x: x**2 - c_val
            root, converged = secant(f, x0, x1)
            return root

        solve_compiled = torch.compile(solve, backend="eager")

        x0 = torch.tensor([1.0, 1.0, 1.0])
        x1 = torch.tensor([2.0, 2.0, 2.0])
        roots = solve_compiled(x0, x1, c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSecantCUDA:
    """Tests for CUDA device support."""

    def test_cuda_basic(self):
        """Test basic functionality on CUDA."""
        device = torch.device("cuda")
        c = torch.tensor([2.0], device=device)
        f = lambda x: x**2 - c
        x0 = torch.tensor([1.5], device=device)

        root, converged = secant(f, x0)

        assert root.device.type == "cuda"
        assert converged.device.type == "cuda"
        expected = math.sqrt(2)
        torch.testing.assert_close(
            root.cpu(), torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_cuda_batched(self):
        """Test batched operation on CUDA."""
        device = torch.device("cuda")
        c = torch.tensor([2.0, 3.0, 4.0, 5.0], device=device)
        f = lambda x: x**2 - c
        x0 = torch.full((4,), 1.5, device=device)

        roots, converged = secant(f, x0)

        assert roots.device.type == "cuda"
        assert converged.device.type == "cuda"
        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_cuda_autograd(self):
        """Test autograd on CUDA."""
        device = torch.device("cuda")
        theta = torch.tensor(
            [2.0], dtype=torch.float64, device=device, requires_grad=True
        )

        def f(x):
            return x**2 - theta

        x0 = torch.tensor([1.5], dtype=torch.float64, device=device)

        root, converged = secant(f, x0)
        root.sum().backward()

        expected = 1.0 / (2.0 * math.sqrt(2.0))
        torch.testing.assert_close(
            theta.grad.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        assert converged.all()
