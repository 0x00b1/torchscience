# tests/torchscience/root_finding/test__newton_system.py
import math

import pytest
import torch

from torchscience.root_finding import newton_system


class TestNewtonSystem:
    """Tests for Newton-Raphson method for systems of equations."""

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
        root, converged = newton_system(f, x0)

        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-10, atol=1e-10)
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
        root, converged = newton_system(f, x0)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [expected_val, expected_val], dtype=torch.float64
        )
        torch.testing.assert_close(root, expected, rtol=1e-10, atol=1e-10)
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
        roots, converged = newton_system(f, x0)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 3, dtype=torch.float64
        )
        torch.testing.assert_close(roots, expected, rtol=1e-10, atol=1e-10)
        assert converged.all()

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
        root, converged = newton_system(f, x0, jacobian=jacobian)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]], dtype=torch.float64
        )
        torch.testing.assert_close(root, expected, rtol=1e-10, atol=1e-10)
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
        root, converged = newton_system(f, x0)

        # Should return unbatched output: shape (2,)
        assert root.shape == (2,)
        assert converged.shape == ()  # Scalar bool

        # Solution: x1 = 1, x2 = 2
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-10, atol=1e-10)
        assert converged.item()

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
        root, converged = newton_system(f, x0)

        expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_batched_2d_systems(self):
        """Solve multiple 2D linear systems in parallel with explicit Jacobian.

        When solving different systems (with different RHS), we need to use
        an explicit Jacobian since autodiff through vmap doesn't handle
        batch-varying captured tensors correctly.
        """
        # Different linear systems with batch-varying RHS
        sums = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float64)
        diffs = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64)

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1 + x2 - sums
            f2 = x1 - x2 - diffs
            return torch.stack([f1, f2], dim=-1)

        def jacobian(x):
            # J = [[1, 1], [1, -1]] - constant for all batch elements
            batch_size = x.shape[0]
            J = torch.zeros(batch_size, 2, 2, dtype=x.dtype, device=x.device)
            J[..., 0, 0] = 1
            J[..., 0, 1] = 1
            J[..., 1, 0] = 1
            J[..., 1, 1] = -1
            return J

        x0 = torch.zeros((3, 2), dtype=torch.float64)
        roots, converged = newton_system(f, x0, jacobian=jacobian)

        expected = torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float64
        )
        torch.testing.assert_close(roots, expected, rtol=1e-10, atol=1e-10)
        assert converged.all()

    def test_rosenbrock_system(self):
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
        root, converged = newton_system(f, x0, maxiter=100)

        expected = torch.tensor([1.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_scalar_system(self):
        """Works with n=1 (effectively a scalar function)."""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([1.5], dtype=torch.float64)
        root, converged = newton_system(f, x0)

        expected = torch.tensor([math.sqrt(2)], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-10, atol=1e-10)
        assert converged.item()

    def test_non_convergence(self):
        """Returns converged=False when not converging within maxiter."""

        def f(x):
            # A system that's hard to solve from this starting point
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1**2 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.0, 0.0], dtype=torch.float64)

        # With very few iterations, shouldn't converge
        root, converged = newton_system(f, x0, maxiter=1)

        # Should return something finite
        assert torch.isfinite(root).all()
        # Check shape is preserved
        assert root.shape == (2,)

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""

        def f(x):
            return x**2 - 2

        for dtype in [torch.float32, torch.float64]:
            x0 = torch.tensor([1.5], dtype=dtype)
            root, converged = newton_system(f, x0)

            assert root.dtype == dtype

    def test_explicit_vs_autodiff_same_result(self):
        """Explicit Jacobian gives same result as autodiff."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        def jacobian(x):
            batch_size = x.shape[0]
            x1, x2 = x[..., 0], x[..., 1]
            J = torch.zeros(batch_size, 2, 2, dtype=x.dtype, device=x.device)
            J[..., 0, 0] = 2 * x1
            J[..., 0, 1] = 2 * x2
            J[..., 1, 0] = 1
            J[..., 1, 1] = -1
            return J

        x0 = torch.tensor([[0.5, 0.5]], dtype=torch.float64)

        root_autodiff, conv_autodiff = newton_system(f, x0)
        root_explicit, conv_explicit = newton_system(f, x0, jacobian=jacobian)

        torch.testing.assert_close(
            root_autodiff, root_explicit, rtol=1e-10, atol=1e-10
        )
        assert conv_autodiff.all() == conv_explicit.all()

    def test_empty_batch(self):
        """Handle empty batch gracefully."""

        def f(x):
            return x**2 - 2

        x0 = torch.zeros((0, 2), dtype=torch.float64)
        root, converged = newton_system(f, x0)

        assert root.shape == (0, 2)
        assert converged.shape == (0,)


class TestNewtonSystemVmap:
    """Tests for vmap compatibility.

    Note: vmap is currently NOT compatible with newton_system due to:
    1. Internal use of requires_grad_() for autodiff (not supported in vmap)
    2. Data-dependent control flow (if torch.all(converged)) not supported
    3. torch.linalg.solve operations inside vmap

    These tests document the expected incompatibility. Use explicit batching
    (e.g., newton_system(f, batched_x0)) instead of vmap for vectorized computation.
    """

    @pytest.mark.xfail(
        reason="vmap incompatible: requires_grad_() and data-dependent control flow"
    )
    def test_vmap_basic(self):
        """vmap works with newton_system for vectorized parameter sweeps."""
        from torch.func import vmap

        theta = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        def solve_one(t):
            def f(x):
                x1, x2 = x[..., 0], x[..., 1]
                f1 = x1**2 + x2**2 - t
                f2 = x1 - x2
                return torch.stack([f1, f2], dim=-1)

            x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
            root, converged = newton_system(f, x0)
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

            root, converged = newton_system(f, x0)
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

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_with_explicit_jacobian(self):
        """vmap works with explicit Jacobian."""
        from torch.func import vmap

        theta = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

        def solve_one(t):
            def f(x):
                x1, x2 = x[..., 0], x[..., 1]
                f1 = x1**2 + x2**2 - t
                f2 = x1 - x2
                return torch.stack([f1, f2], dim=-1)

            def jacobian(x):
                batch_size = x.shape[0]
                x1, x2 = x[..., 0], x[..., 1]
                J = torch.zeros(
                    batch_size, 2, 2, dtype=x.dtype, device=x.device
                )
                J[..., 0, 0] = 2 * x1
                J[..., 0, 1] = 2 * x2
                J[..., 1, 0] = 1
                J[..., 1, 1] = -1
                return J

            x0 = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
            root, converged = newton_system(f, x0, jacobian=jacobian)
            return root.squeeze(0)

        # Use vmap to vectorize over the first dimension
        roots = vmap(solve_one)(theta)

        # Root is (sqrt(theta/2), sqrt(theta/2))
        expected_vals = torch.sqrt(theta / 2)
        expected = torch.stack([expected_vals, expected_vals], dim=-1)
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)

    def test_explicit_batching_alternative(self):
        """Demonstrates explicit batching as alternative to vmap.

        Instead of using vmap, pass batched inputs directly to newton_system.
        """

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1.0
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor(
            [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]], dtype=torch.float64
        )

        roots, converged = newton_system(f, x0)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 3, dtype=torch.float64
        )
        torch.testing.assert_close(roots, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestNewtonSystemCUDA:
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
        root, converged = newton_system(f, x0)

        assert root.device.type == "cuda"
        assert converged.device.type == "cuda"

        expected = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
        torch.testing.assert_close(root, expected, rtol=1e-10, atol=1e-10)
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
        roots, converged = newton_system(f, x0)

        assert roots.device.type == "cuda"
        assert converged.device.type == "cuda"

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 3,
            dtype=torch.float64,
            device=device,
        )
        torch.testing.assert_close(roots, expected, rtol=1e-10, atol=1e-10)
        assert converged.all()
