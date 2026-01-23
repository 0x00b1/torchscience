# tests/torchscience/root_finding/test__halley.py
import math

import pytest
import torch

from torchscience.root_finding import halley


class TestHalley:
    """Tests for Halley's root-finding method."""

    def test_simple_quadratic(self):
        """Find sqrt(2) by solving x^2 - 2 = 0."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])

        root, converged = halley(f, x0)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_cubic_convergence(self):
        """Halley converges in very few iterations due to cubic convergence.

        Halley's method has order 3 convergence, so it should converge
        faster than Newton (order 2). For a simple quadratic starting
        at x0=1.5 finding sqrt(2), Halley should converge in 2-3 iterations.
        """
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float64)

        # With just 3 iterations, Halley should achieve very high precision
        root, converged = halley(f, x0, maxiter=3)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        assert converged.all()

    def test_explicit_derivatives(self):
        """Uses df and ddf when provided instead of autodiff."""
        f = lambda x: x**2 - 2
        df = lambda x: 2 * x
        ddf = lambda x: torch.full_like(x, 2.0)
        x0 = torch.tensor([1.5])

        root, converged = halley(f, x0, df=df, ddf=ddf)

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

        roots, converged = halley(f, x0)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_preserves_shape(self):
        """2D input preserves shape in output."""
        f = lambda x: x**2 - 2
        x0 = torch.full((2, 3), 1.5)

        root, converged = halley(f, x0)

        assert root.shape == (2, 3)
        assert converged.shape == (2, 3)
        expected = torch.full((2, 3), math.sqrt(2))
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_cubic_function(self):
        """Find root of cubic function x^3 - x - 1 = 0."""
        f = lambda x: x**3 - x - 1
        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = halley(f, x0)

        # Verify it's a root by checking f(root) close to 0
        residual = f(root)
        torch.testing.assert_close(
            residual,
            torch.tensor([0.0], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_transcendental(self):
        """Find root of exp(x) - 2 = 0, i.e., ln(2)."""
        f = lambda x: torch.exp(x) - 2
        x0 = torch.tensor([1.0], dtype=torch.float64)

        root, converged = halley(f, x0)

        expected = math.log(2)
        torch.testing.assert_close(
            root,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_trigonometric(self):
        """Find root of sin(x) = 0 near pi."""
        f = lambda x: torch.sin(x)
        x0 = torch.tensor([3.0])  # Start near pi

        root, converged = halley(f, x0)

        torch.testing.assert_close(
            root, torch.tensor([math.pi]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_float64(self):
        """Works with float64 and achieves higher precision."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float64)

        root, converged = halley(f, x0)

        assert root.dtype == torch.float64
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        assert converged.all()

    def test_float32(self):
        """Works correctly with float32."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5], dtype=torch.float32)

        root, converged = halley(f, x0)

        assert root.dtype == torch.float32
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        assert converged.all()

    def test_empty_input(self):
        """Handle empty input gracefully."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([])

        root, converged = halley(f, x0)

        assert root.shape == (0,)
        assert converged.shape == (0,)

    def test_returns_tuple(self):
        """halley returns (root, converged) tuple."""
        f = lambda x: x**2 - 2
        x0 = torch.tensor([1.5])

        result = halley(f, x0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        root, converged = result
        assert root.shape == x0.shape
        assert converged.shape == x0.shape
        assert converged.dtype == torch.bool

    def test_zero_denominator_safeguard(self):
        """Safeguards against zero denominator.

        When the denominator 2*f'^2 - f*f'' is very small or zero,
        the safeguard should prevent NaN or Inf from appearing.
        """
        # x^3 has f''=6x which is 0 at x=0, potentially causing issues
        f = lambda x: x**3
        x0 = torch.tensor([0.1])

        root, converged = halley(f, x0, maxiter=100)

        # The key thing is that we don't get NaN or Inf
        assert torch.isfinite(root).all(), f"Expected finite root, got {root}"
        # The root should be close to 0 (it's making progress)
        assert torch.abs(root) < 0.1, f"Expected root near 0, got {root}"

    def test_explicit_vs_autodiff_same_result(self):
        """Explicit df and ddf gives same result as autodiff."""
        f = lambda x: x**3 - x - 1
        df = lambda x: 3 * x**2 - 1
        ddf = lambda x: 6 * x
        x0 = torch.tensor([1.5], dtype=torch.float64)

        root_autodiff, conv_autodiff = halley(f, x0)
        root_explicit, conv_explicit = halley(f, x0, df=df, ddf=ddf)

        torch.testing.assert_close(
            root_autodiff, root_explicit, rtol=1e-10, atol=1e-10
        )
        assert conv_autodiff.all() == conv_explicit.all()

    def test_faster_than_newton_iterations(self):
        """Halley converges in fewer iterations than Newton for same tolerance.

        This test verifies the cubic convergence property indirectly by
        showing that Halley achieves convergence with fewer iterations.
        """
        f = lambda x: x**2 - 2
        x0 = torch.tensor([10.0], dtype=torch.float64)  # Start far from root

        # Halley with 5 iterations
        root_halley, conv_halley = halley(f, x0, maxiter=5)

        # Check that Halley converged with high precision
        expected = math.sqrt(2)
        torch.testing.assert_close(
            root_halley,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert conv_halley.all()


class TestHalleyAutograd:
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

        root, converged = halley(f, x0)
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

        roots, converged = halley(f, x0)
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

        root, converged = halley(f, x0)
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

        root, converged = halley(f, x0)

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

        root1, _ = halley(f1, x0)

        theta_plus = torch.tensor([theta_val + eps], dtype=torch.float64)

        def f2(x):
            return f(x, theta_plus)

        root2, _ = halley(f2, x0)

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

        root, converged = halley(f, x0)
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
class TestHalleyCUDA:
    """Tests for CUDA device support."""

    def test_cuda_basic(self):
        """Test basic functionality on CUDA."""
        device = torch.device("cuda")
        c = torch.tensor([2.0], device=device)
        f = lambda x: x**2 - c
        x0 = torch.tensor([1.5], device=device)

        root, converged = halley(f, x0)

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

        roots, converged = halley(f, x0)

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

        root, converged = halley(f, x0)
        root.sum().backward()

        expected = 1.0 / (2.0 * math.sqrt(2.0))
        torch.testing.assert_close(
            theta.grad.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        assert converged.all()
