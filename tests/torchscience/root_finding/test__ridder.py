# tests/torchscience/root_finding/test__ridder.py
import math

import pytest
import torch

from torchscience.root_finding import ridder

# Check if scipy is available for comparison tests
try:
    from scipy.optimize import ridder as scipy_ridder

    scipy_available = True
except ImportError:
    scipy_available = False


class TestRidder:
    """Tests for Ridder's root-finding method."""

    def test_simple_quadratic(self):
        """Find sqrt(2) by solving x^2 - 2 = 0."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root, converged = ridder(f, a, b)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            root, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_batched(self):
        """Find multiple roots in parallel."""
        c = torch.tensor([2.0, 3.0, 4.0, 5.0])
        f = lambda x: x**2 - c
        a = torch.ones(4)
        b = torch.full((4,), 10.0)

        roots, converged = ridder(f, a, b)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_trigonometric(self):
        """Find root of sin(x) = 0 in [2, 4] -> pi."""
        f = lambda x: torch.sin(x)
        a = torch.tensor([2.0])
        b = torch.tensor([4.0])

        root, converged = ridder(f, a, b)

        torch.testing.assert_close(
            root, torch.tensor([math.pi]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_invalid_bracket_raises(self):
        """Raise ValueError when f(a) and f(b) have same sign."""
        f = lambda x: x**2 + 1  # Always positive
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        with pytest.raises(ValueError, match="Invalid bracket"):
            ridder(f, a, b)

    def test_preserves_shape(self):
        """Output has same shape as input and correct values."""
        f = lambda x: x**2 - 2
        a = torch.ones(2, 3)
        b = torch.full((2, 3), 2.0)

        root, converged = ridder(f, a, b)

        assert root.shape == (2, 3)
        assert converged.shape == (2, 3)
        expected = torch.full((2, 3), math.sqrt(2))
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy(self):
        """Results match scipy.optimize.ridder."""
        f_torch = lambda x: x**2 - 2
        f_scipy = lambda x: x**2 - 2

        scipy_root = scipy_ridder(f_scipy, 1.0, 2.0)
        our_root, converged = ridder(
            f_torch,
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
        )

        torch.testing.assert_close(
            our_root,
            torch.tensor([scipy_root], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

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

        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        root, converged = ridder(f, a, b)
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

    def test_root_at_endpoint_a(self):
        """Return a immediately if f(a) == 0."""
        f = lambda x: x - 1.0
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root, converged = ridder(f, a, b)

        torch.testing.assert_close(root, torch.tensor([1.0]))
        assert converged.all()

    def test_root_at_endpoint_b(self):
        """Return b immediately if f(b) == 0."""
        f = lambda x: x - 2.0
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root, converged = ridder(f, a, b)

        torch.testing.assert_close(root, torch.tensor([2.0]))
        assert converged.all()

    def test_shape_mismatch_raises(self):
        """Raise ValueError when a and b have different shapes."""
        f = lambda x: x
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0])

        with pytest.raises(ValueError, match="must have same shape"):
            ridder(f, a, b)

    def test_nan_input_raises(self):
        """Raise ValueError when inputs contain NaN."""
        f = lambda x: x
        a = torch.tensor([float("nan")])
        b = torch.tensor([1.0])

        with pytest.raises(ValueError, match="must not contain NaN"):
            ridder(f, a, b)

    def test_maxiter_exceeded_returns_best_estimate(self):
        """Return best estimate with converged=False when maxiter is exceeded."""
        f = lambda x: x**3 - x - 1
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        root, converged = ridder(f, a, b, maxiter=1)

        # Should not converge with only 1 iteration
        assert not converged.all()
        # But should still return a valid estimate within bounds
        assert root.shape == a.shape
        assert (root >= a).all() and (root <= b).all()

    def test_float32(self):
        """Works correctly with float32."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float32)
        b = torch.tensor([2.0], dtype=torch.float32)

        root, converged = ridder(f, a, b)

        assert root.dtype == torch.float32
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        assert converged.all()

    def test_float64(self):
        """Works correctly with float64."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        root, converged = ridder(f, a, b)

        assert root.dtype == torch.float64
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_empty_input(self):
        """Handle empty input gracefully."""
        f = lambda x: x
        a = torch.tensor([])
        b = torch.tensor([])

        root, converged = ridder(f, a, b)

        assert root.shape == (0,)
        assert converged.shape == (0,)

    def test_returns_tuple(self):
        """ridder returns (root, converged) tuple."""
        f = lambda x: x**2 - 2
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])

        result = ridder(f, a, b)

        assert isinstance(result, tuple)
        assert len(result) == 2
        root, converged = result
        assert root.shape == a.shape
        assert converged.shape == a.shape
        assert converged.dtype == torch.bool

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy_cubic(self):
        """Results match scipy.optimize.ridder for cubic function."""
        f_torch = lambda x: x**3 - x - 1
        f_scipy = lambda x: x**3 - x - 1

        scipy_root = scipy_ridder(f_scipy, 1.0, 2.0)
        our_root, converged = ridder(
            f_torch,
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
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
        """Results match scipy.optimize.ridder for transcendental function."""
        f_torch = lambda x: torch.exp(x) - 2
        f_scipy = lambda x: math.exp(x) - 2

        scipy_root = scipy_ridder(f_scipy, 0.0, 1.0)
        our_root, converged = ridder(
            f_torch,
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        )

        torch.testing.assert_close(
            our_root,
            torch.tensor([scipy_root], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy_trigonometric(self):
        """Results match scipy.optimize.ridder for trigonometric function."""
        f_torch = lambda x: torch.sin(x)
        f_scipy = lambda x: math.sin(x)

        scipy_root = scipy_ridder(f_scipy, 2.0, 4.0)
        our_root, converged = ridder(
            f_torch,
            torch.tensor([2.0], dtype=torch.float64),
            torch.tensor([4.0], dtype=torch.float64),
        )

        torch.testing.assert_close(
            our_root,
            torch.tensor([scipy_root], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()


class TestRidderAutograd:
    """Tests for autograd support via implicit differentiation."""

    def test_implicit_diff_batched(self):
        """Test gradient with batched inputs."""
        theta = torch.tensor(
            [2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True
        )

        def f(x):
            return x**2 - theta

        a = torch.zeros(3, dtype=torch.float64)
        b = torch.full((3,), 10.0, dtype=torch.float64)

        roots, converged = ridder(f, a, b)
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

        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([10.0], dtype=torch.float64)

        root, converged = ridder(f, a, b)
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

        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)

        root, converged = ridder(f, a, b)

        # Should just return the root without error
        torch.testing.assert_close(
            root,
            torch.tensor([math.sqrt(2)], dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        assert converged.all()

    def test_gradient_with_loss_function(self):
        """Test gradient computation through a loss function."""
        theta = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([1.5], dtype=torch.float64)

        def f(x):
            return x**2 - theta

        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([10.0], dtype=torch.float64)

        root, converged = ridder(f, a, b)
        loss = (root - target) ** 2  # MSE loss

        loss.backward()

        # root = sqrt(theta) = 2.0
        # d(loss)/d(root) = 2 * (root - target) = 2 * (2.0 - 1.5) = 1.0
        # d(root)/d(theta) = 1/(2*sqrt(theta)) = 1/4 = 0.25
        # d(loss)/d(theta) = 1.0 * 0.25 = 0.25
        expected = torch.tensor([0.25], dtype=torch.float64)
        torch.testing.assert_close(theta.grad, expected, rtol=1e-4, atol=1e-6)
        assert converged.all()

    def test_implicit_diff_numerical_verification(self):
        """Verify gradient numerically (finite differences)."""
        theta_val = 2.0
        eps = 1e-5

        def f(x, t):
            return x**2 - t

        a = torch.tensor([0.0], dtype=torch.float64)
        b = torch.tensor([3.0], dtype=torch.float64)

        # Compute root at theta and theta + eps
        theta = torch.tensor(
            [theta_val], dtype=torch.float64, requires_grad=True
        )

        def f1(x):
            return f(x, theta)

        root1, _ = ridder(f1, a, b)

        theta_plus = torch.tensor([theta_val + eps], dtype=torch.float64)

        def f2(x):
            return f(x, theta_plus)

        root2, _ = ridder(f2, a, b)

        # Numerical gradient
        numerical_grad = (root2 - root1) / eps

        # Analytical gradient via backward
        root1.sum().backward()
        analytical_grad = theta.grad

        torch.testing.assert_close(
            analytical_grad, numerical_grad, rtol=1e-3, atol=1e-6
        )

    def test_gradgradcheck(self):
        """Test second-order gradients via gradgradcheck.

        For f(x) = x^2 - theta, root x* = sqrt(theta).
        First derivative: dx*/dtheta = 1/(2*sqrt(theta))
        Second derivative: d^2x*/dtheta^2 = -1/(4*theta^(3/2))
        """

        def func(theta):
            f = lambda x: x**2 - theta
            a = torch.tensor([0.5], dtype=torch.float64)
            b = torch.tensor([3.0], dtype=torch.float64)
            root, _ = ridder(f, a, b)
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
            a = torch.tensor([0.5], dtype=torch.float64)
            b = torch.tensor([3.0], dtype=torch.float64)
            root, _ = ridder(f, a, b)
            return root.sum()

        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(func, theta)


class TestRidderVmap:
    """Tests for vmap compatibility.

    Note: vmap is currently NOT compatible with ridder due to:
    1. Data-dependent control flow (if torch.all(converged)) not supported

    These tests document the expected incompatibility. Use explicit batching
    (e.g., ridder(f, batched_a, batched_b)) instead of vmap for vectorized computation.
    """

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_basic(self):
        """vmap works with ridder for vectorized parameter sweeps."""
        from torch.func import vmap

        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        def solve_one(ci):
            f = lambda x: x**2 - ci
            a = torch.tensor([1.0], dtype=torch.float64)
            b = torch.tensor([10.0], dtype=torch.float64)
            root, converged = ridder(f, a, b)
            return root

        # Use vmap to vectorize over the first dimension
        roots = vmap(solve_one)(c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(
            roots.squeeze(-1), expected, rtol=1e-6, atol=1e-6
        )

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_different_brackets(self):
        """vmap works with different brackets for the same problem."""
        from torch.func import vmap

        def solve_sqrt2(a_scalar):
            a = a_scalar.unsqueeze(0)
            b = torch.tensor([3.0], dtype=torch.float64)
            f = lambda x: x**2 - 2.0
            root, converged = ridder(f, a, b)
            return root

        a_vals = torch.tensor([0.5, 1.0, 1.2], dtype=torch.float64)
        roots = vmap(solve_sqrt2)(a_vals)

        expected = torch.full((3, 1), math.sqrt(2), dtype=torch.float64)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_with_grad(self):
        """vmap + grad works together for parameter gradients."""
        from torch.func import grad, vmap

        def solve_and_return(ci):
            f = lambda x: x**2 - ci
            a = torch.tensor([0.5], dtype=torch.float64)
            b = torch.tensor([10.0], dtype=torch.float64)
            root, _ = ridder(f, a, b)
            return root.sum()

        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        # grad of vmapped function
        grads = vmap(grad(solve_and_return))(c)

        # d(sqrt(c))/dc = 1/(2*sqrt(c))
        expected = 1.0 / (2.0 * torch.sqrt(c))
        torch.testing.assert_close(grads, expected, rtol=1e-4, atol=1e-6)

    def test_explicit_batching_alternative(self):
        """Demonstrates explicit batching as alternative to vmap.

        Instead of using vmap, pass batched inputs directly to ridder.
        """
        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        f = lambda x: x**2 - c
        a = torch.ones(3, dtype=torch.float64)
        b = torch.full((3,), 10.0, dtype=torch.float64)

        roots, converged = ridder(f, a, b)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()


class TestRidderCompile:
    """Tests for torch.compile compatibility.

    Note: torch.compile may have issues with data-dependent loops used in
    iterative root-finding algorithms. These tests verify compatibility
    and document any limitations.
    """

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_eager_backend(self):
        """torch.compile with eager backend works with ridder."""
        c = torch.tensor([2.0, 3.0, 4.0])

        def solve(a, b, c_val):
            f = lambda x: x**2 - c_val
            root, converged = ridder(f, a, b)
            return root

        # Compile with eager backend (more forgiving)
        solve_compiled = torch.compile(solve, backend="eager")

        a = torch.ones(3)
        b = torch.full((3,), 10.0)
        roots = solve_compiled(a, b, c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(roots, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(
        reason="torch.compile with inductor may not support data-dependent loops"
    )
    def test_compile_inductor_backend(self):
        """torch.compile with inductor backend works with ridder."""
        c = torch.tensor([2.0, 3.0, 4.0])

        def solve(a, b, c_val):
            f = lambda x: x**2 - c_val
            root, converged = ridder(f, a, b)
            return root

        # Compile with default inductor backend
        solve_compiled = torch.compile(solve)

        a = torch.ones(3)
        b = torch.full((3,), 10.0)
        roots = solve_compiled(a, b, c)

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
            a = torch.tensor([0.0], dtype=torch.float64)
            b = torch.tensor([2.0], dtype=torch.float64)
            root, _ = ridder(f, a, b)
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
    def test_compile_trigonometric(self):
        """torch.compile works with trigonometric function."""

        def solve(a, b):
            f = lambda x: torch.sin(x)
            root, converged = ridder(f, a, b)
            return root

        solve_compiled = torch.compile(solve, backend="eager")

        a = torch.tensor([2.0])
        b = torch.tensor([4.0])
        root = solve_compiled(a, b)

        torch.testing.assert_close(
            root, torch.tensor([math.pi]), rtol=1e-5, atol=1e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestRidderCUDA:
    """Tests for CUDA device support."""

    def test_cuda_basic(self):
        """Test basic functionality on CUDA."""
        device = torch.device("cuda")
        c = torch.tensor([2.0], device=device)
        f = lambda x: x**2 - c
        a = torch.tensor([1.0], device=device)
        b = torch.tensor([2.0], device=device)

        root, converged = ridder(f, a, b)

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
        a = torch.ones(4, device=device)
        b = torch.full((4,), 10.0, device=device)

        roots, converged = ridder(f, a, b)

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

        a = torch.tensor([0.0], dtype=torch.float64, device=device)
        b = torch.tensor([2.0], dtype=torch.float64, device=device)

        root, converged = ridder(f, a, b)
        root.sum().backward()

        expected = 1.0 / (2.0 * math.sqrt(2.0))
        torch.testing.assert_close(
            theta.grad.cpu(),
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        assert converged.all()
