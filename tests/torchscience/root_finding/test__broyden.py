# tests/torchscience/root_finding/test__broyden.py
"""Tests for Broyden's quasi-Newton method."""

import math

import pytest
import torch

from torchscience.root_finding import broyden


class TestBroyden:
    """Tests for Broyden's quasi-Newton method for systems of equations."""

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
        root, converged = broyden(f, x0)

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
        root, converged = broyden(f, x0)

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
        roots, converged = broyden(f, x0)

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
        root, converged = broyden(f, x0)

        # Should return unbatched output: shape (2,)
        assert root.shape == (2,)
        assert converged.shape == ()  # Scalar bool

        # Solution: x1 = 1, x2 = 2
        expected = torch.tensor([1.0, 2.0], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_good_method(self):
        """Test Broyden's 'good' (first) method explicitly."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2 - 1
            f2 = x1 + x2**2 - 1
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
        root, converged = broyden(f, x0, method="good")

        # The system has solutions at (0, 1) and (1, 0)
        # From (0.5, 0.5), should converge to one of them
        # Check that f(root) is close to zero
        fx = f(root)
        assert torch.allclose(fx, torch.zeros_like(fx), atol=1e-8)
        assert converged.item()

    def test_bad_method(self):
        """Test Broyden's 'bad' (second) method explicitly."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2 - 1
            f2 = x1 + x2**2 - 1
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)
        root, converged = broyden(f, x0, method="bad")

        # Check that f(root) is close to zero
        fx = f(root)
        assert torch.allclose(fx, torch.zeros_like(fx), atol=1e-8)
        assert converged.item()

    def test_explicit_jacobian_init(self):
        """Uses explicit initial Jacobian approximation when provided."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)

        # Provide an approximation to the Jacobian at x0
        # Exact Jacobian at (0.5, 0.5) is [[1, 1], [1, -1]]
        J_init = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)

        root, converged = broyden(f, x0, jacobian_init=J_init)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [expected_val, expected_val], dtype=torch.float64
        )
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_batched_jacobian_init(self):
        """Uses batched initial Jacobian approximation."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([[0.5, 0.5], [0.6, 0.6]], dtype=torch.float64)

        # Batched initial Jacobians (approximate)
        J_init = torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[1.2, 1.2], [1.0, -1.0]]],
            dtype=torch.float64,
        )

        roots, converged = broyden(f, x0, jacobian_init=J_init)

        expected_val = 1.0 / math.sqrt(2)
        expected = torch.tensor(
            [[expected_val, expected_val]] * 2, dtype=torch.float64
        )
        torch.testing.assert_close(roots, expected, rtol=1e-8, atol=1e-8)
        assert converged.all()

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
        root, converged = broyden(f, x0)

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
        root, converged = broyden(f, x0, maxiter=2)

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
            root, converged = broyden(f, x0)

            assert root.dtype == dtype

    def test_empty_batch(self):
        """Handle empty batch gracefully."""

        def f(x):
            return x**2 - 2

        x0 = torch.zeros((0, 2), dtype=torch.float64)
        root, converged = broyden(f, x0)

        assert root.shape == (0, 2)
        assert converged.shape == (0,)

    def test_scalar_system(self):
        """Works with n=1 (effectively a scalar function)."""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([1.5], dtype=torch.float64)
        root, converged = broyden(f, x0)

        expected = torch.tensor([math.sqrt(2)], dtype=torch.float64)
        torch.testing.assert_close(root, expected, rtol=1e-8, atol=1e-8)
        assert converged.item()

    def test_invalid_method(self):
        """Raises ValueError for invalid method."""

        def f(x):
            return x**2 - 2

        x0 = torch.tensor([1.5, 1.5], dtype=torch.float64)

        with pytest.raises(ValueError, match="method"):
            broyden(f, x0, method="invalid")

    def test_good_vs_bad_both_converge(self):
        """Both 'good' and 'bad' methods should converge to same solution."""

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64)

        root_good, conv_good = broyden(f, x0, method="good")
        root_bad, conv_bad = broyden(f, x0, method="bad")

        assert conv_good.item() and conv_bad.item()

        # Both should converge to the same root
        torch.testing.assert_close(root_good, root_bad, rtol=1e-6, atol=1e-6)


class TestBroydenComparison:
    """Compare Broyden to scipy.optimize.broyden1/broyden2."""

    @pytest.mark.parametrize("method", ["good", "bad"])
    def test_compare_to_scipy(self, method):
        """Verify results match scipy.optimize.broyden1/broyden2."""
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
        root_torch, converged = broyden(f_torch, x0_torch, method=method)
        assert converged.item()

        # scipy broyden1 is "good", broyden2 is "bad"
        if method == "good":
            root_scipy = scipy_optimize.broyden1(f_numpy, x0, f_tol=1e-10)
        else:
            root_scipy = scipy_optimize.broyden2(f_numpy, x0, f_tol=1e-10)

        torch.testing.assert_close(
            root_torch,
            torch.tensor(root_scipy, dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBroydenCUDA:
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
        root, converged = broyden(f, x0)

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
        roots, converged = broyden(f, x0)

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

    def test_cuda_jacobian_init(self):
        """Test with initial Jacobian on CUDA."""
        device = torch.device("cuda")

        def f(x):
            x1, x2 = x[..., 0], x[..., 1]
            f1 = x1**2 + x2**2 - 1
            f2 = x1 - x2
            return torch.stack([f1, f2], dim=-1)

        x0 = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
        J_init = torch.eye(2, dtype=torch.float64, device=device)

        root, converged = broyden(f, x0, jacobian_init=J_init)

        assert root.device.type == "cuda"
        assert converged.item()
