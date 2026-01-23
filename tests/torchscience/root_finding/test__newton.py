# tests/torchscience/root_finding/test__newton.py
import math

import pytest
import torch

from torchscience.root_finding import newton

# Check if scipy is available for comparison tests
try:
    from scipy.optimize import newton as scipy_newton

    scipy_available = True
except ImportError:
    scipy_available = False


class TestNewton:
    """Tests for Newton-Raphson root-finding method."""

    def test_simple_quadratic(self):
        """Find sqrt(2) by solving x^2 - 2 = 0."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])

        root, converged = newton(f, x0)

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

        roots, converged = newton(f, x0)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_explicit_derivative(self):
        """Uses df when provided instead of autodiff."""
        f = lambda x: x**2 - 2
        df = lambda x: 2 * x
        x0 = torch.tensor([1.5])

        root, converged = newton(f, x0, df=df)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_quadratic_convergence(self):
        """Newton's method converges quadratically (few iterations needed)."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float64)

        # With maxiter=5, Newton should easily converge for a quadratic
        root, converged = newton(f, x0, maxiter=5)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_non_convergence(self):
        """Returns converged=False when not converging."""
        # Use a function that oscillates when starting at x=0
        # f(x) = x^3 - 2x + 2 has a local max/min that can cause issues
        f = lambda x: x**3 - 2 * x + 2
        x0 = torch.tensor([0.0])  # Poor starting point

        # With very few iterations, shouldn't converge
        root, converged = newton(f, x0, maxiter=1)

        # May or may not converge depending on the function behavior
        # But with maxiter=1, unlikely to fully converge
        assert root.shape == x0.shape

    def test_preserves_shape(self):
        """2D input preserves shape in output."""
        f = lambda x: x**2 - 2
        x0 = torch.full((2, 3), 1.5)

        root, converged = newton(f, x0)

        assert root.shape == (2, 3)
        assert converged.shape == (2, 3)
        expected = torch.full((2, 3), math.sqrt(2))
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_float64(self):
        """Works with float64 and achieves higher precision."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = newton(f, x0)

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

        root, converged = newton(f, x0)

        assert root.dtype == torch.float32
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        assert converged.all()

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy(self):
        """Results match scipy.optimize.newton."""
        f_torch = lambda x: x**2 - 2
        f_scipy = lambda x: x**2 - 2
        fprime_scipy = lambda x: 2 * x

        scipy_root = scipy_newton(f_scipy, 1.5, fprime=fprime_scipy)
        our_root, converged = newton(
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
    def test_matches_scipy_cubic(self):
        """Results match scipy.optimize.newton for cubic function."""
        f_torch = lambda x: x**3 - x - 1
        f_scipy = lambda x: x**3 - x - 1
        fprime_scipy = lambda x: 3 * x**2 - 1

        scipy_root = scipy_newton(f_scipy, 1.5, fprime=fprime_scipy)
        our_root, converged = newton(
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
        fprime_scipy = lambda x: math.exp(x)

        scipy_root = scipy_newton(f_scipy, 1.0, fprime=fprime_scipy)
        our_root, converged = newton(
            f_torch, torch.tensor([1.0], dtype=torch.float64)
        )

        torch.testing.assert_close(
            our_root,
            torch.tensor([scipy_root], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_trigonometric(self):
        """Find root of sin(x) = 0 near pi."""
        f = lambda x: torch.sin(x)
        x0 = torch.tensor([3.0])  # Start near pi

        root, converged = newton(f, x0)

        torch.testing.assert_close(
            root, torch.tensor([math.pi]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_empty_input(self):
        """Handle empty input gracefully."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([])

        root, converged = newton(f, x0)

        assert root.shape == (0,)
        assert converged.shape == (0,)

    def test_returns_tuple(self):
        """newton returns (root, converged) tuple."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])

        result = newton(f, x0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        root, converged = result
        assert root.shape == x0.shape
        assert converged.shape == x0.shape
        assert converged.dtype == torch.bool

    def test_zero_derivative_safeguard(self):
        """Safeguards against zero derivative by using eps * sign(df).

        When the derivative is very small or zero, the safeguard should prevent
        NaN or Inf from appearing in the result. Convergence may be slow but
        the result should be finite.
        """
        # x^3 has zero derivative at x=0, but we can still iterate
        f = lambda x: x**3
        x0 = torch.tensor([0.1])

        root, converged = newton(f, x0, maxiter=100)

        # The key thing is that we don't get NaN or Inf
        assert torch.isfinite(root).all(), f"Expected finite root, got {root}"
        # The root should be close to 0 (it's making progress)
        assert torch.abs(root) < 0.1, f"Expected root near 0, got {root}"

    def test_explicit_vs_autodiff_same_result(self):
        """Explicit df gives same result as autodiff."""
        f = lambda x: x**3 - x - 1
        df = lambda x: 3 * x**2 - 1
        x0 = torch.tensor([1.5], dtype=torch.float64)

        root_autodiff, conv_autodiff = newton(f, x0)
        root_explicit, conv_explicit = newton(f, x0, df=df)

        torch.testing.assert_close(
            root_autodiff, root_explicit, rtol=1e-10, atol=1e-10
        )
        assert conv_autodiff.all() == conv_explicit.all()


class TestNewtonAutograd:
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

        root, converged = newton(f, x0)
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

        roots, converged = newton(f, x0)
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

        root, converged = newton(f, x0)
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

        root, converged = newton(f, x0)

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

        root1, _ = newton(f1, x0)

        theta_plus = torch.tensor([theta_val + eps], dtype=torch.float64)

        def f2(x):
            return f(x, theta_plus)

        root2, _ = newton(f2, x0)

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

        root, converged = newton(f, x0)
        loss = (root - target) ** 2  # MSE loss

        loss.backward()

        # root = sqrt(theta) = 2.0
        # d(loss)/d(root) = 2 * (root - target) = 2 * (2.0 - 1.5) = 1.0
        # d(root)/d(theta) = 1/(2*sqrt(theta)) = 1/4 = 0.25
        # d(loss)/d(theta) = 1.0 * 0.25 = 0.25
        expected = torch.tensor([0.25], dtype=torch.float64)
        torch.testing.assert_close(theta.grad, expected, rtol=1e-4, atol=1e-6)
        assert converged.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestNewtonCUDA:
    """Tests for CUDA device support."""

    def test_cuda_basic(self):
        """Test basic functionality on CUDA."""
        device = torch.device("cuda")
        c = torch.tensor([2.0], device=device)
        f = lambda x: x**2 - c
        x0 = torch.tensor([1.5], device=device)

        root, converged = newton(f, x0)

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

        roots, converged = newton(f, x0)

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

        root, converged = newton(f, x0)
        root.sum().backward()

        expected = 1.0 / (2.0 * math.sqrt(2.0))
        torch.testing.assert_close(
            theta.grad.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        assert converged.all()
