# tests/torchscience/root_finding/test__fixed_point.py
import math

import pytest
import torch

from torchscience.root_finding import fixed_point

# Check if scipy is available for comparison tests
try:
    from scipy.optimize import fixed_point as scipy_fixed_point

    scipy_available = True
except ImportError:
    scipy_available = False


class TestFixedPoint:
    """Tests for fixed-point iteration method."""

    def test_simple_contraction(self):
        """Solve x = cos(x) (Dottie number)."""
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([1.0])

        fp, converged = fixed_point(g, x0)

        # Dottie number is approximately 0.739085
        expected = 0.7390851332151607
        torch.testing.assert_close(
            fp, torch.tensor([expected]), rtol=1e-5, atol=1e-5
        )
        assert converged.all()

    def test_sqrt_iteration(self):
        """Solve x = (x + 2/x) / 2 to find sqrt(2) (Babylonian method)."""
        g = lambda x: (x + 2 / x) / 2
        x0 = torch.tensor([1.5])

        fp, converged = fixed_point(g, x0)

        expected = math.sqrt(2)
        torch.testing.assert_close(
            fp, torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_batched(self):
        """Batched fixed-point iteration for sqrt(c)."""
        c = torch.tensor([2.0, 3.0, 4.0, 5.0])
        g = lambda x: (x + c / x) / 2  # Babylonian method
        x0 = torch.full((4,), 1.5)

        fps, converged = fixed_point(g, x0)

        expected = torch.sqrt(c)
        torch.testing.assert_close(fps, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_system(self):
        """Fixed-point for 2D system."""
        # Simple contractive system:
        # x1_new = 0.5 * x2
        # x2_new = 0.5 * x1 + 0.2
        # Fixed point: x1 = 0.1, x2 = 0.25 (solving the system)
        # Actually: x1 = 0.5 * x2, x2 = 0.5 * x1 + 0.2
        # => x1 = 0.5 * (0.5 * x1 + 0.2) = 0.25 * x1 + 0.1
        # => 0.75 * x1 = 0.1 => x1 = 0.1/0.75 = 2/15
        # => x2 = 0.5 * (2/15) + 0.2 = 1/15 + 0.2 = 1/15 + 3/15 = 4/15

        def g(x):
            x1, x2 = x[..., 0], x[..., 1]
            return torch.stack([0.5 * x2, 0.5 * x1 + 0.2], dim=-1)

        x0 = torch.tensor([[1.0, 1.0]])  # Shape (1, 2)

        fp, converged = fixed_point(g, x0)

        # Expected fixed point
        expected_x1 = 2 / 15
        expected_x2 = 4 / 15
        expected = torch.tensor([[expected_x1, expected_x2]])

        assert fp.shape == (1, 2)
        assert converged.shape == (1,)
        torch.testing.assert_close(fp, expected, rtol=1e-5, atol=1e-5)
        assert converged.all()

    def test_preserves_shape(self):
        """Output shape matches input for scalar problems."""
        g = lambda x: torch.cos(x)
        x0 = torch.full((2, 3), 1.0)

        fp, converged = fixed_point(g, x0)

        # For 2D input, it's treated as a system with batch shape (2,) and system dim 3
        # So converged has shape (2,)
        assert fp.shape == (2, 3)
        assert converged.shape == (2,)

    def test_preserves_shape_1d(self):
        """Output shape matches input for 1D tensor."""
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([1.0, 0.5, 0.8])

        fp, converged = fixed_point(g, x0)

        assert fp.shape == (3,)
        assert converged.shape == (3,)
        assert converged.all()

    def test_float64(self):
        """Works with float64 and achieves higher precision."""
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([1.0], dtype=torch.float64)

        fp, converged = fixed_point(g, x0)

        assert fp.dtype == torch.float64
        # Dottie number to high precision
        expected = 0.7390851332151606416553120876738734040134
        torch.testing.assert_close(
            fp,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-9,
            atol=1e-9,
        )
        assert converged.all()

    def test_float32(self):
        """Works correctly with float32."""
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([1.0], dtype=torch.float32)

        fp, converged = fixed_point(g, x0)

        assert fp.dtype == torch.float32
        expected = 0.7390851
        torch.testing.assert_close(
            fp,
            torch.tensor([expected], dtype=torch.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        assert converged.all()

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy(self):
        """Results match scipy.optimize.fixed_point for cos iteration."""
        g_torch = lambda x: torch.cos(x)
        g_scipy = lambda x: math.cos(x)

        scipy_fp = float(scipy_fixed_point(g_scipy, 1.0))
        our_fp, converged = fixed_point(
            g_torch, torch.tensor([1.0], dtype=torch.float64)
        )

        torch.testing.assert_close(
            our_fp,
            torch.tensor([scipy_fp], dtype=torch.float64),
            rtol=1e-9,
            atol=1e-9,
        )
        assert converged.all()

    @pytest.mark.skipif(not scipy_available, reason="scipy not available")
    def test_matches_scipy_sqrt(self):
        """Results match scipy.optimize.fixed_point for sqrt iteration."""
        g_torch = lambda x: (x + 2 / x) / 2
        g_scipy = lambda x: (x + 2 / x) / 2

        scipy_fp = float(scipy_fixed_point(g_scipy, 1.5))
        our_fp, converged = fixed_point(
            g_torch, torch.tensor([1.5], dtype=torch.float64)
        )

        torch.testing.assert_close(
            our_fp,
            torch.tensor([scipy_fp], dtype=torch.float64),
            rtol=1e-9,
            atol=1e-9,
        )
        assert converged.all()

    def test_non_convergence(self):
        """Returns converged=False when not converging."""
        # Non-contractive function (expands rather than contracts)
        g = lambda x: 2 * x
        x0 = torch.tensor([1.0])

        fp, converged = fixed_point(g, x0, maxiter=10)

        # Should not converge for this function
        assert not converged.all()
        assert fp.shape == x0.shape

    def test_empty_input(self):
        """Handle empty input gracefully."""
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([])

        fp, converged = fixed_point(g, x0)

        assert fp.shape == (0,)
        assert converged.shape == (0,)

    def test_returns_tuple(self):
        """fixed_point returns (fixed_point, converged) tuple."""
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([1.0])

        result = fixed_point(g, x0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        fp, converged = result
        assert fp.shape == x0.shape
        assert converged.shape == x0.shape
        assert converged.dtype == torch.bool

    def test_cubic_root(self):
        """Find cube root of 2 using fixed-point iteration."""
        # x = (2 * x + 2 / x^2) / 3 converges to cube root of 2
        g = lambda x: (2 * x + 2 / x**2) / 3
        x0 = torch.tensor([1.5], dtype=torch.float64)

        fp, converged = fixed_point(g, x0)

        expected = 2 ** (1 / 3)
        torch.testing.assert_close(
            fp,
            torch.tensor([expected], dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10,
        )
        assert converged.all()

    def test_linear_convergence(self):
        """Test convergence for a simple linear contraction."""
        # x = 0.5 * x + 1 has fixed point x = 2
        g = lambda x: 0.5 * x + 1
        x0 = torch.tensor([0.0])

        fp, converged = fixed_point(g, x0)

        torch.testing.assert_close(
            fp, torch.tensor([2.0]), rtol=1e-5, atol=1e-5
        )
        assert converged.all()

    def test_batched_system(self):
        """Test batched system with shape (B, n)."""

        def g(x):
            # Simple contraction for each batch element
            return 0.5 * x + 0.25

        x0 = torch.tensor([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])  # Shape (3, 2)

        fp, converged = fixed_point(g, x0)

        # Fixed point: x = 0.5 * x + 0.25 => 0.5 * x = 0.25 => x = 0.5
        expected = torch.full((3, 2), 0.5)
        assert fp.shape == (3, 2)
        assert converged.shape == (3,)
        torch.testing.assert_close(fp, expected, rtol=1e-5, atol=1e-5)
        assert converged.all()

    def test_custom_tolerance(self):
        """Test with custom tolerance parameters."""
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([1.0])

        # Very loose tolerance should converge quickly
        fp, converged = fixed_point(g, x0, xtol=0.1, rtol=0.1, maxiter=5)
        assert converged.all()

        # Very tight tolerance with few iterations may not converge
        fp2, converged2 = fixed_point(g, x0, xtol=1e-15, rtol=1e-15, maxiter=5)
        # May or may not converge depending on iterations

    def test_different_starting_points(self):
        """Same function with different starting points converges to same fixed point."""
        g = lambda x: torch.cos(x)
        x0_vals = [0.5, 1.0, 1.5, 2.0]

        results = []
        for val in x0_vals:
            x0 = torch.tensor([val], dtype=torch.float64)
            fp, converged = fixed_point(g, x0)
            assert converged.all()
            results.append(fp)

        # All should converge to the same fixed point
        for r in results[1:]:
            torch.testing.assert_close(results[0], r, rtol=1e-8, atol=1e-8)


class TestFixedPointVmap:
    """Tests for vmap compatibility.

    Note: vmap is currently NOT compatible with fixed_point due to:
    1. Data-dependent control flow (if torch.all(converged)) not supported

    These tests document the expected incompatibility. Use explicit batching
    (e.g., fixed_point(g, batched_x0)) instead of vmap for vectorized computation.
    """

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_basic(self):
        """vmap works with fixed_point for vectorized parameter sweeps."""
        from torch.func import vmap

        # Find sqrt(c) using Babylonian method
        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

        def solve_one(ci):
            g = lambda x: (x + ci / x) / 2
            x0 = torch.tensor([1.5], dtype=torch.float64)
            fp, converged = fixed_point(g, x0)
            return fp

        # Use vmap to vectorize over the first dimension
        fps = vmap(solve_one)(c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(
            fps.squeeze(-1), expected, rtol=1e-6, atol=1e-6
        )

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_different_starting_points(self):
        """vmap works with different starting points for the same problem."""
        from torch.func import vmap

        def solve_dottie(x0_scalar):
            x0 = x0_scalar.unsqueeze(0)
            g = lambda x: torch.cos(x)
            fp, converged = fixed_point(g, x0)
            return fp

        x0_vals = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
        fps = vmap(solve_dottie)(x0_vals)

        # Dottie number
        expected = torch.full((3, 1), 0.7390851332151607, dtype=torch.float64)
        torch.testing.assert_close(fps, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.xfail(reason="vmap incompatible: data-dependent control flow")
    def test_vmap_system(self):
        """vmap works with fixed-point systems."""
        from torch.func import vmap

        k = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float64)

        def solve_one(ki):
            def g(x):
                # Simple contraction: x_new = k * x + (1 - k)
                # Fixed point is 1.0 for any k in (0, 1)
                return ki * x + (1 - ki)

            x0 = torch.tensor([0.0], dtype=torch.float64)
            fp, converged = fixed_point(g, x0)
            return fp

        fps = vmap(solve_one)(k)

        expected = torch.ones(3, 1, dtype=torch.float64)
        torch.testing.assert_close(fps, expected, rtol=1e-6, atol=1e-6)

    def test_explicit_batching_alternative(self):
        """Demonstrates explicit batching as alternative to vmap.

        Instead of using vmap, pass batched inputs directly to fixed_point.
        """
        c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        g = lambda x: (x + c / x) / 2  # Babylonian method
        x0 = torch.full((3,), 1.5, dtype=torch.float64)

        fps, converged = fixed_point(g, x0)

        expected = torch.sqrt(c)
        torch.testing.assert_close(fps, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()


class TestFixedPointCompile:
    """Tests for torch.compile compatibility.

    Note: torch.compile may have issues with data-dependent loops used in
    iterative root-finding algorithms. These tests verify compatibility
    and document any limitations.
    """

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_eager_backend(self):
        """torch.compile with eager backend works with fixed_point."""
        c = torch.tensor([2.0, 3.0, 4.0])

        def solve(x0, c_val):
            g = lambda x: (x + c_val / x) / 2  # Babylonian method
            fp, converged = fixed_point(g, x0)
            return fp

        # Compile with eager backend (more forgiving)
        solve_compiled = torch.compile(solve, backend="eager")

        x0 = torch.tensor([1.5, 1.5, 1.5])
        fps = solve_compiled(x0, c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(fps, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    @pytest.mark.xfail(
        reason="torch.compile with inductor may not support data-dependent loops"
    )
    def test_compile_inductor_backend(self):
        """torch.compile with inductor backend works with fixed_point."""
        c = torch.tensor([2.0, 3.0, 4.0])

        def solve(x0, c_val):
            g = lambda x: (x + c_val / x) / 2  # Babylonian method
            fp, converged = fixed_point(g, x0)
            return fp

        # Compile with default inductor backend
        solve_compiled = torch.compile(solve)

        x0 = torch.tensor([1.5, 1.5, 1.5])
        fps = solve_compiled(x0, c)

        expected = torch.sqrt(c)
        torch.testing.assert_close(fps, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_dottie_number(self):
        """torch.compile works for finding the Dottie number."""

        def solve(x0):
            g = lambda x: torch.cos(x)
            fp, converged = fixed_point(g, x0)
            return fp

        solve_compiled = torch.compile(solve, backend="eager")

        x0 = torch.tensor([1.0])
        fp = solve_compiled(x0)

        # Dottie number is approximately 0.739085
        expected = torch.tensor([0.7390851332151607])
        torch.testing.assert_close(fp, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_system(self):
        """torch.compile works with fixed-point systems."""

        def solve(x0):
            def g(x):
                x1, x2 = x[..., 0], x[..., 1]
                return torch.stack([0.5 * x2, 0.5 * x1 + 0.2], dim=-1)

            fp, converged = fixed_point(g, x0)
            return fp

        solve_compiled = torch.compile(solve, backend="eager")

        x0 = torch.tensor([[1.0, 1.0]])
        fp = solve_compiled(x0)

        expected_x1 = 2 / 15
        expected_x2 = 4 / 15
        expected = torch.tensor([[expected_x1, expected_x2]])
        torch.testing.assert_close(fp, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(
        not hasattr(torch, "compile"), reason="torch.compile not available"
    )
    def test_compile_batched(self):
        """torch.compile works with batched inputs."""

        def solve(x0):
            g = lambda x: 0.5 * x + 0.25
            fp, converged = fixed_point(g, x0)
            return fp

        solve_compiled = torch.compile(solve, backend="eager")

        x0 = torch.tensor([[1.0, 2.0], [0.0, 1.0], [2.0, 0.0]])
        fps = solve_compiled(x0)

        # Fixed point: x = 0.5 * x + 0.25 => x = 0.5
        expected = torch.full((3, 2), 0.5)
        torch.testing.assert_close(fps, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFixedPointCUDA:
    """Tests for CUDA device support."""

    def test_cuda_basic(self):
        """Test basic functionality on CUDA."""
        device = torch.device("cuda")
        g = lambda x: torch.cos(x)
        x0 = torch.tensor([1.0], device=device)

        fp, converged = fixed_point(g, x0)

        assert fp.device.type == "cuda"
        assert converged.device.type == "cuda"
        expected = 0.7390851
        torch.testing.assert_close(
            fp.cpu(), torch.tensor([expected]), rtol=1e-6, atol=1e-6
        )
        assert converged.all()

    def test_cuda_batched(self):
        """Test batched operation on CUDA."""
        device = torch.device("cuda")
        c = torch.tensor([2.0, 3.0, 4.0, 5.0], device=device)
        g = lambda x: (x + c / x) / 2
        x0 = torch.full((4,), 1.5, device=device)

        fps, converged = fixed_point(g, x0)

        assert fps.device.type == "cuda"
        assert converged.device.type == "cuda"
        expected = torch.sqrt(c)
        torch.testing.assert_close(fps, expected, rtol=1e-6, atol=1e-6)
        assert converged.all()

    def test_cuda_system(self):
        """Test system on CUDA."""
        device = torch.device("cuda")

        def g(x):
            x1, x2 = x[..., 0], x[..., 1]
            return torch.stack([0.5 * x2, 0.5 * x1 + 0.2], dim=-1)

        x0 = torch.tensor([[1.0, 1.0]], device=device)

        fp, converged = fixed_point(g, x0)

        assert fp.device.type == "cuda"
        assert converged.device.type == "cuda"
        assert fp.shape == (1, 2)
        assert converged.all()
