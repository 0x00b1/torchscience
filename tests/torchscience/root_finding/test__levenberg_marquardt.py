# tests/torchscience/root_finding/test__levenberg_marquardt.py
"""Tests for Levenberg-Marquardt algorithm."""

import math

import pytest
import torch

from torchscience.root_finding import levenberg_marquardt


class TestLevenbergMarquardt:
    """Tests for Levenberg-Marquardt algorithm for systems of equations."""

    def test_linear_system(self):
        """Solve a linear system Ax = b.

        System:
            x1 + x2 = 3
            2*x1 - x2 = 0

        Solution: x1 = 1, x2 = 2
        """

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1 + x2 - 3  # x + y = 3
            f2 = 2 * x1 - x2  # 2x - y = 0
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0)

        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_nonlinear_system(self):
        """Solve a nonlinear system.

        System:
            x^2 + y^2 = 1  (unit circle)
            x = y

        Solution: x = y = 1/sqrt(2) (for positive values)
        """

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1  # x^2 + y^2 = 1
            f2 = x1 - x2  # x = y
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [expected_val, expected_val], dtype=torch.float64
        )
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_batched(self):
        """Solve multiple 2D systems in parallel with same function.

        Solve x^2 + y^2 = 1, x = y from different starting points.
        All should converge to (1/sqrt(2), 1/sqrt(2)).
        """

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        # Batch of 3 starting points, all positive quadrant
        x0 = torch.tensor(
            [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]], dtype=torch.float64
        )
        roots, converged = levenberg_marquardt(f, x0)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 3, dtype=torch.float64
        )
        torch.testing.assert_close(roots, expected, rtol=1e-8, atol=1e-8)
        assert converged.all()

    def test_unbatched_input(self):
        """Works with unbatched (n,) input."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1 + x2 - 3
            f2 = x1 - x2 + 1
            return torch.stack([f1, f2], dim=-1)

        # Unbatched input: shape (2,)
        x0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0)

        # Should return unbatched output: shape (2,)
        assert root.shape == (2,)
        assert converged.shape == ()  # Scalar bool

        # Solution: x1 = 1, x2 = 2
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_explicit_jacobian(self):
        """Uses explicit Jacobian when provided."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        def jacobian(x):
            # J = [[2*x1, 2*x2], [1, -1]]
            batch_size = x.shape[0]
            x1, x2 = x[..., 0], x[..., 1]
            J = torch.zeros(batch_size, 2, 2, dtype=x.dtype, device=x.device)
            J[..., 0, 0] = 2 * x1
            J[..., 0, 1] = 2 * x2
            J[..., 1, 0] = 1
            J[..., 1, 1] = -1
            return J

        x0 = torch.tensor([[0.5, 0.5]], dtype=torch.float64)  # Batched input
        root, converged = levenberg_marquardt(f, x0, jacobian=jacobian)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]], dtype=torch.float64
        )
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.all()

    def test_damping_parameter(self):
        """Test that damping parameter affects convergence behavior."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)

        # Test with different damping values
        root_low_damp, conv_low = levenberg_marquardt(f, x0, damping=1e-6)
        root_high_damp, conv_high = levenberg_marquardt(f, x0, damping=1.0)

        # Both should converge to the same solution
        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [expected_val, expected_val], dtype=torch.float64
        )

        torch.testing.assert_close(
            root_low_damp, expected, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            root_high_damp, expected, rtol=1e-6, atol=1e-6
        )
        assert conv_low.item() and conv_high.item()

    def test_3d_system(self):
        """Solve a 3D nonlinear system.

        System:
            x + y + z = 6
            x*y*z = 6
            x + y - z = 0

        Solution: x = 1, y = 2, z = 3 (one of the solutions)
        """

        def f(x):
            x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
            f1 = x1 + x2 + x3 - 6
            f2 = x1 * x2 * x3 - 6
            f3 = x1 + x2 - x3
            return torch.stack([f1, f2, f3], dim=-1)

        # Starting point close to expected solution (1, 2, 3)
        x0 = torch.tensor([1.1, 1.9, 3.1], dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0)

        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)
        assert converged.item()

    def test_non_convergence(self):
        """Returns converged=False when not converging within maxiter."""

        def f(x):
            # A system that's hard to solve from this starting point
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1**2 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.01, 0.01], dtype=torch.float64)

        # With very few iterations, shouldn't converge
        root, converged = levenberg_marquardt(f, x0, maxiter=2)

        # Should return something finite
        assert torch.isfinite(root).all()
        # Check shape is preserved
        assert root.shape == (2,)

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""

        def f(x):
            return x**2 - torch.tensor(
                [2.0, 3.0], dtype=x.dtype, device=x.device
            )

        for dtype in [torch.float32, torch.float64]:
            x0 = torch.tensor([1.5, 1.5], dtype=dtype)
            root, converged = levenberg_marquardt(f, x0)

            assert root.dtype == dtype

    def test_empty_batch(self):
        """Handle empty batch gracefully."""

        def f(x):
            return x**2 - 2

        x0 = torch.zeros((0, 2), dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0)

        assert root.shape == (0, 2)
        assert converged.shape == (0,)

    def test_scalar_system(self):
        """Works with n=1 (effectively a scalar function)."""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([1.5], dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0)

        expected = torch.tensor([math.sqrt(2)], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_rosenbrock_stationary_point(self):
        """Find stationary point of Rosenbrock function (gradient = 0).

        Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        Gradient = 0 at (1, 1).
        """

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            # Gradient of Rosenbrock
            df_dx = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
            df_dy = 200 * (x2 - x1**2)
            return torch.stack([df_dx, df_dy], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0, maxiter=200)

        expected = torch.tensor([1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)
        assert converged.item()

    def test_overdetermined_system(self):
        """Test with overdetermined system (more equations than unknowns).

        Levenberg-Marquardt is naturally suited for least-squares problems.
        """

        def f(x):
            # 3 equations, 2 unknowns
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1 + x2 - 2
            f2 = x1 - x2
            f3 = x1 + 2 * x2 - 3  # Consistent with f1 and f2
            return torch.stack([f1, f2, f3], dim=-1)

        x0 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        root, converged = levenberg_marquardt(f, x0)

        # System is consistent: solution is (1, 1)
        expected = torch.tensor([1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)


class TestLevenbergMarquardtAutograd:
    """Tests for autograd support via implicit differentiation."""

    def test_implicit_diff_2d_system(self):
        """Test gradient w.r.t. function parameter for 2D system.

        For f(x, y) = [x^2 + y^2 - theta, x - y], root is x* = y* = sqrt(theta/2).
        By implicit function theorem: d(x*)/d(theta) = -J^{-1} @ df/dtheta
        """
        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - theta
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([1.5, 1.5], dtype=torch.float64)

        root, converged = levenberg_marquardt(f, x0)
        loss = root.sum()
        loss.backward()

        # At theta=2, root is (1, 1)
        # d(loss)/dtheta = 0.5 (same calculation as in test__broyden.py)
        expected = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(theta.grad, expected, rtol=1e-4, atol=1e-6)
        assert converged.item()

    def test_implicit_diff_batched(self):
        """Test gradient with batched inputs.

        Note: When using batched inputs with captured parameters that vary
        per batch element, we need to use explicit Jacobian to avoid vmap
        issues with captured tensors.
        """
        theta = torch.tensor(
            [[1.0], [2.0], [4.0]], dtype=torch.float64, requires_grad=True
        )

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - theta.squeeze(-1)
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        def jacobian(x):
            # J = [[2*x1, 2*x2], [1, -1]]
            batch_size = x.shape[0]
            x1, x2 = x[..., 0], x[..., 1]
            J = torch.zeros(batch_size, 2, 2, dtype=x.dtype, device=x.device)
            J[..., 0, 0] = 2 * x1
            J[..., 0, 1] = 2 * x2
            J[..., 1, 0] = 1
            J[..., 1, 1] = -1
            return J

        x0 = torch.tensor(
            [[1.0, 1.0], [1.2, 1.2], [1.5, 1.5]], dtype=torch.float64
        )

        roots, converged = levenberg_marquardt(f, x0, jacobian=jacobian)
        loss = roots.sum()
        loss.backward()

        expected = 1.0 / torch.sqrt(2.0 * theta.detach())
        torch.testing.assert_close(theta.grad, expected, rtol=1e-3, atol=1e-5)
        assert converged.all()

    def test_implicit_diff_linear_system(self):
        """Test gradient with linear parameter dependence.

        System: x + y = theta, x - y = 0
        Root: x = y = theta/2
        d(x+y)/dtheta = 1
        """
        theta = torch.tensor([6.0], dtype=torch.float64, requires_grad=True)

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1 + x2 - theta
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([2.0, 2.0], dtype=torch.float64)

        root, converged = levenberg_marquardt(f, x0)
        loss = root.sum()
        loss.backward()

        expected = torch.tensor([1.0], dtype=torch.float64)
        torch.testing.assert_close(theta.grad, expected, rtol=1e-4, atol=1e-6)
        assert converged.item()

    def test_implicit_diff_no_param_grad(self):
        """Test that no error occurs when function has no differentiable parameters."""
        constant = 1.0  # Not a tensor with requires_grad

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - constant
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)

        root, converged = levenberg_marquardt(f, x0)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [expected_val, expected_val], dtype=torch.float64
        )
        torch.testing.assert_close(root, expected, rtol=1e-6, atol=1e-6)
        assert converged.item()

    def test_implicit_diff_numerical_verification(self):
        """Verify gradient numerically (finite differences)."""
        theta_val = 2.0
        eps = 1e-5

        def make_f(t):
            def f(x):
                x1, x2 = x[..., 0], x[..., 1]
                f1 = x1**2 + x2**2 - t
                f2 = x1 - x2
                return torch.stack([f1, f2], dim=-1)

            return f

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)

        # Compute roots at theta and theta + eps
        theta = torch.tensor(
            [theta_val], dtype=torch.float64, requires_grad=True
        )
        root1, _ = levenberg_marquardt(make_f(theta), x0)

        theta_plus = torch.tensor([theta_val + eps], dtype=torch.float64)
        root2, _ = levenberg_marquardt(make_f(theta_plus), x0)

        # Numerical gradient of sum(root) w.r.t. theta
        numerical_grad = (root2.sum() - root1.sum()) / eps

        # Analytical gradient via backward
        root1.sum().backward()
        analytical_grad = theta.grad

        torch.testing.assert_close(
            analytical_grad,
            numerical_grad.unsqueeze(0),
            rtol=1e-3,
            atol=1e-5,
        )

    def test_gradient_with_loss_function(self):
        """Test gradient computation through a loss function."""
        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0.5, 0.5], dtype=torch.float64)

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - theta
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([1.5, 1.5], dtype=torch.float64)

        root, converged = levenberg_marquardt(f, x0)
        loss = ((root - target) ** 2).sum()  # MSE loss

        loss.backward()

        expected = torch.tensor([0.5], dtype=torch.float64)
        torch.testing.assert_close(theta.grad, expected, rtol=1e-3, atol=1e-5)
        assert converged.item()

    def test_gradgradcheck(self):
        """Test second-order gradients via gradgradcheck.

        For 2D system f(x,y) = [x^2 + y^2 - theta, x - y],
        root is x* = y* = sqrt(theta/2), so x* + y* = sqrt(2*theta).
        First derivative: d(x*+y*)/dtheta = 1/sqrt(2*theta)
        Second derivative: d^2(x*+y*)/dtheta^2 = -1/(2*(2*theta)^(3/2))
        """

        def func(theta):
            def f(x):
                x1, x2 = x[..., 0], x[..., 1]
                f1 = x1**2 + x2**2 - theta
                f2 = x1 - x2
                return torch.stack([f1, f2], dim=-1)

            x0 = torch.tensor([1.5, 1.5], dtype=torch.float64)
            root, _ = levenberg_marquardt(f, x0)
            return root.sum()

        theta = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
        torch.autograd.gradgradcheck(func, theta)


class TestLevenbergMarquardtComparison:
    """Compare Levenberg-Marquardt to scipy.optimize.root with 'lm' method."""

    def test_compare_to_scipy(self):
        """Verify results match scipy.optimize.root with Levenberg-Marquardt."""
        scipy_optimize = pytest.importorskip("scipy.optimize")

        def f_torch(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        def f_numpy(x):
            return [x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1]]

        x0 = [0.5, 0.5]
        x0_torch = torch.tensor(x0, dtype=torch.float64)

        # Our implementation
        root_torch, converged = levenberg_marquardt(f_torch, x0_torch)
        assert converged.item()

        # scipy with 'lm' method (Levenberg-Marquardt)
        result = scipy_optimize.root(f_numpy, x0, method="lm")

        torch.testing.assert_close(
            root_torch,
            torch.tensor(result.x, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )


class TestLevenbergMarquardtVmap:
    """Tests for vmap compatibility.

    Note: vmap is currently NOT compatible with levenberg_marquardt due to:
    1. Internal use of requires_grad_() for autodiff (not supported in vmap)
    2. Data-dependent control flow (if torch.all(converged)) not supported
    3. torch.linalg.solve operations inside vmap

    These tests document the expected incompatibility. Use explicit batching
    (e.g., levenberg_marquardt(f, batched_x0)) instead of vmap for vectorized computation.
    """

    @pytest.mark.xfail(
        reason="vmap incompatible: requires_grad_() and data-dependent control flow"
    )
    def test_vmap_basic(self):
        """vmap works with levenberg_marquardt for vectorized parameter sweeps."""
        from torch.func import vmap

        theta = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        def solve_one(t):
            def f(x):
                x1, x2 = x[..., 0], x[..., 1]
                f1 = x1**2 + x2**2 - t
                f2 = x1 - x2
                return torch.stack([f1, f2], dim=-1)

            x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
            root, converged = levenberg_marquardt(f, x0)
            return root

        # Use vmap to vectorize over the first dimension
        roots = vmap(solve_one)(theta)

        # Root is (sqrt(theta/2), sqrt(theta/2))
        expected_vals = torch.sqrt(theta / 2)
        expected = torch.stack([expected_vals, expected_vals], dim=-1)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.xfail(
        reason="vmap incompatible: requires_grad_() and data-dependent control flow"
    )
    def test_vmap_different_starting_points(self):
        """vmap works with different starting points for the same problem."""
        from torch.func import vmap

        def solve_circle(x0):
            def f(x):
                x1, x2 = x[..., 0], x[..., 1]
                f1 = x1**2 + x2**2 - 1.0
                f2 = x1 - x2
                return torch.stack([f1, f2], dim=-1)

            root, converged = levenberg_marquardt(f, x0)
            return root

        x0_vals = torch.tensor(
            [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]], dtype=torch.float64
        )
        roots = vmap(solve_circle)(x0_vals)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 3, dtype=torch.float64
        )
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.xfail(
        reason="vmap incompatible: requires_grad_() and data-dependent control flow"
    )
    def test_vmap_with_grad(self):
        """vmap + grad works together for parameter gradients."""
        from torch.func import grad, vmap

        def solve_and_sum(t):
            def f(x):
                x1, x2 = x[..., 0], x[..., 1]
                f1 = x1**2 + x2**2 - t
                f2 = x1 - x2
                return torch.stack([f1, f2], dim=-1)

            x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
            root, _ = levenberg_marquardt(f, x0)
            return root.sum()

        theta = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        # grad of vmapped function
        grads = vmap(grad(solve_and_sum))(theta)

        # d(x1+x2)/dtheta = 1/sqrt(2*theta) from implicit diff
        expected = 1.0 / torch.sqrt(2.0 * theta)
        torch.testing.assert_close(grads, expected, rtol=1e-3, atol=1e-5)

    def test_explicit_batching_alternative(self):
        """Demonstrates explicit batching as alternative to vmap.

        Instead of using vmap, pass batched inputs directly to levenberg_marquardt.
        """

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1.0
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor(
            [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]], dtype=torch.float64
        )

        roots, converged = levenberg_marquardt(f, x0)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 3, dtype=torch.float64
        )
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestLevenbergMarquardtCUDA:
    """Tests for CUDA device support."""

    def test_cuda_basic(self):
        """Test basic functionality on CUDA."""
        device = torch.device("cuda")

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1 + x2 - 3
            f2 = x1 - x2 + 1
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
        root, converged = levenberg_marquardt(f, x0)

        assert root.device.type == "cuda"
        assert converged.device.type == "cuda"

        expected = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_cuda_batched(self):
        """Test batched operation on CUDA."""
        device = torch.device("cuda")

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor(
            [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]],
            dtype=torch.float64,
            device=device,
        )
        roots, converged = levenberg_marquardt(f, x0)

        assert roots.device.type == "cuda"
        assert converged.device.type == "cuda"

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 3,
            dtype=torch.float64,
            device=device,
        )
        torch.testing.assert_close(roots, expected, rtol=1e-8, atol=1e-8)
        assert converged.all()
