"""Tests for B-spline basis function evaluation."""

import pytest
import torch


class TestBSpline:
    """Tests for BSpline tensorclass."""

    def test_bspline_creation(self):
        """Can create BSpline with valid inputs."""
        from torchscience.spline import BSpline

        # Clamped cubic B-spline with 5 control points
        # n_knots = n_control + degree + 1 = 5 + 3 + 1 = 9
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [[0.0, 0.0], [0.25, 0.5], [0.5, 1.0], [0.75, 0.5], [1.0, 0.0]],
            dtype=torch.float64,
        )
        degree = 3
        extrapolate = "error"

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=degree,
            extrapolate=extrapolate,
            batch_size=[],
        )

        assert isinstance(spline, BSpline)
        assert spline.knots.shape == (9,)
        assert spline.control_points.shape == (5, 2)
        assert spline.degree == 3
        assert spline.extrapolate == "error"


class TestBSplineEvaluate:
    """Tests for b_spline_evaluate function."""

    def test_evaluate_linear(self):
        """Linear B-spline (degree 1) interpolates control points.

        For degree 1, with clamped knots, the spline passes through
        all control points at the corresponding knots.
        """
        from torchscience.spline import BSpline, b_spline_evaluate

        # Degree 1 with 3 control points
        # n_knots = n_control + degree + 1 = 3 + 1 + 1 = 5
        # Clamped: [0, 0, 0.5, 1, 1]
        knots = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0], dtype=torch.float64)
        control_points = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=1,
            extrapolate="error",
            batch_size=[],
        )

        # Evaluate at knot positions (interior knots)
        t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = b_spline_evaluate(spline, t)

        # For clamped linear spline, should pass through control points
        expected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(y, expected, atol=1e-10, rtol=1e-10)

    def test_evaluate_cubic(self):
        """Cubic B-spline (degree 3) is smooth."""
        from torchscience.spline import BSpline, b_spline_evaluate

        # Clamped cubic B-spline
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [0.0, 1.0, 1.0, 0.0], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Evaluate at multiple points
        t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        y = b_spline_evaluate(spline, t)

        # Check shape
        assert y.shape == (50,)

        # Check smoothness: values should vary continuously
        # The curve should be bell-shaped (starts at 0, goes to ~0.67, back to 0)
        assert y[0].item() == pytest.approx(0.0, abs=1e-10)
        assert y[-1].item() == pytest.approx(0.0, abs=1e-10)
        assert y[25].item() > 0.5  # Peak should be positive

    def test_evaluate_at_endpoints(self):
        """Clamped knots: spline passes through first/last control points."""
        from torchscience.spline import BSpline, b_spline_evaluate

        # Clamped cubic B-spline with 5 control points
        # n_knots = 5 + 3 + 1 = 9
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [1.0, 2.0, 3.0, 2.0, 1.5], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Evaluate at endpoints
        t = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y = b_spline_evaluate(spline, t)

        # For clamped B-splines, endpoints equal first/last control points
        torch.testing.assert_close(
            y[0], control_points[0], atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            y[1], control_points[-1], atol=1e-10, rtol=1e-10
        )

    def test_evaluate_extrapolate_error(self):
        """Out-of-domain raises ExtrapolationError."""
        from torchscience.spline import (
            BSpline,
            ExtrapolationError,
            b_spline_evaluate,
        )

        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [0.0, 1.0, 1.0, 0.0], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Query outside domain should raise
        with pytest.raises(ExtrapolationError):
            b_spline_evaluate(
                spline, torch.tensor([-0.1], dtype=torch.float64)
            )

        with pytest.raises(ExtrapolationError):
            b_spline_evaluate(spline, torch.tensor([1.1], dtype=torch.float64))

    def test_evaluate_extrapolate_clamp(self):
        """Clamp mode works."""
        from torchscience.spline import BSpline, b_spline_evaluate

        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [1.0, 2.0, 2.0, 3.0], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="clamp",
            batch_size=[],
        )

        # Query outside domain should clamp to boundary values
        t_outside = torch.tensor([-1.0, 2.0], dtype=torch.float64)
        y = b_spline_evaluate(spline, t_outside)

        # Should equal endpoint values
        y_at_0 = b_spline_evaluate(
            spline, torch.tensor([0.0], dtype=torch.float64)
        )
        y_at_1 = b_spline_evaluate(
            spline, torch.tensor([1.0], dtype=torch.float64)
        )

        torch.testing.assert_close(
            y[0], y_at_0.squeeze(), atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            y[1], y_at_1.squeeze(), atol=1e-10, rtol=1e-10
        )

    def test_evaluate_multidimensional(self):
        """Multi-dimensional control points work."""
        from torchscience.spline import BSpline, b_spline_evaluate

        # 2D curve in 3D space
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        # 4 control points, each in 3D
        control_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [2.0, 2.0, 1.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Evaluate at multiple points
        t = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
        y = b_spline_evaluate(spline, t)

        # Output shape should be (10, 3)
        assert y.shape == (10, 3)

        # Endpoints should equal first/last control points (clamped)
        torch.testing.assert_close(
            y[0], control_points[0], atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            y[-1], control_points[-1], atol=1e-10, rtol=1e-10
        )

    def test_scipy_comparison(self):
        """Compare with scipy.interpolate.BSpline."""
        scipy = pytest.importorskip("scipy")
        from scipy.interpolate import BSpline as ScipyBSpline

        from torchscience.spline import BSpline, b_spline_evaluate

        # Clamped cubic B-spline
        knots_np = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
        control_np = [0.0, 0.5, 1.0, 0.8, 0.3, 0.1, 0.5]
        degree = 3

        knots = torch.tensor(knots_np, dtype=torch.float64)
        control_points = torch.tensor(control_np, dtype=torch.float64)

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=degree,
            extrapolate="error",
            batch_size=[],
        )

        # Create scipy equivalent
        scipy_spline = ScipyBSpline(knots_np, control_np, degree)

        # Evaluate at many points
        t = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
        y_torch = b_spline_evaluate(spline, t)
        y_scipy = scipy_spline(t.numpy())

        torch.testing.assert_close(
            y_torch,
            torch.tensor(y_scipy, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_evaluate_scalar_input(self):
        """Scalar input returns scalar output."""
        from torchscience.spline import BSpline, b_spline_evaluate

        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [0.0, 1.0, 1.0, 0.0], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Scalar input
        t = torch.tensor(0.5, dtype=torch.float64)
        y = b_spline_evaluate(spline, t)

        # Should be scalar (0-d tensor)
        assert y.dim() == 0

    def test_evaluate_batch_query_shape(self):
        """Arbitrary query shapes are preserved."""
        from torchscience.spline import BSpline, b_spline_evaluate

        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [0.0, 1.0, 1.0, 0.0], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # 2D query shape
        t = torch.linspace(0.0, 1.0, 12, dtype=torch.float64).reshape(3, 4)
        y = b_spline_evaluate(spline, t)

        # Output shape should match query shape
        assert y.shape == (3, 4)

    def test_gradcheck(self):
        """Verify that gradients work correctly."""
        from torch.autograd import gradcheck

        from torchscience.spline import BSpline, b_spline_evaluate

        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [1.0, 2.0, 3.0, 2.0, 1.5], dtype=torch.float64, requires_grad=True
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Query points (avoid knots where gradient may be discontinuous)
        t = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)

        def eval_fn(cp):
            s = BSpline(
                knots=knots,
                control_points=cp,
                degree=3,
                extrapolate="error",
                batch_size=[],
            )
            return b_spline_evaluate(s, t)

        assert gradcheck(eval_fn, (control_points,), eps=1e-6, atol=1e-4)


class TestBSplineBasis:
    """Tests for b_spline_basis function with Cox-de Boor recursion."""

    def test_degree_0_indicator(self):
        """Test that degree 0 basis functions are indicator functions.

        B_{i,0}(t) = 1 if t_i <= t < t_{i+1}, else 0
        """
        from torchscience.spline import b_spline_basis

        # Uniform knots: [0, 1, 2, 3, 4]
        knots = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        degree = 0

        # n_basis = n_knots - degree - 1 = 5 - 0 - 1 = 4
        # Basis 0: [0, 1), Basis 1: [1, 2), Basis 2: [2, 3), Basis 3: [3, 4]

        # Test points in each interval
        t = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float64)
        basis = b_spline_basis(t, knots, degree)

        # Expected: each point should be 1 in its interval's basis, 0 elsewhere
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],  # t=0.5 in [0,1)
                [0.0, 1.0, 0.0, 0.0],  # t=1.5 in [1,2)
                [0.0, 0.0, 1.0, 0.0],  # t=2.5 in [2,3)
                [0.0, 0.0, 0.0, 1.0],  # t=3.5 in [3,4]
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(basis, expected)

    def test_degree_1_hat(self):
        """Test that degree 1 basis functions are 'hat' (triangular) functions.

        Degree 1 B-splines are piecewise linear with triangular shape.
        """
        from torchscience.spline import b_spline_basis

        # Uniform knots: [0, 1, 2, 3, 4, 5]
        knots = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )
        degree = 1

        # n_basis = 6 - 1 - 1 = 4
        # Basis i has support [t_i, t_{i+2}]

        # Test at knot points (peaks of hat functions)
        t_at_knots = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        basis_at_knots = b_spline_basis(t_at_knots, knots, degree)

        # At t=1: B_0 peaks at 1, others are 0 or 1
        # For degree 1 with uniform knots, B_i(t_{i+1}) = 1
        assert basis_at_knots[0, 0].item() == pytest.approx(1.0)

        # Test at midpoints (half-way up the hat)
        t_mid = torch.tensor([0.5, 1.5], dtype=torch.float64)
        basis_mid = b_spline_basis(t_mid, knots, degree)

        # At t=0.5: B_0 = 0.5 (rising), B_1 = 0 (not started)
        assert basis_mid[0, 0].item() == pytest.approx(0.5)

        # At t=1.5: B_0 = 0.5 (falling), B_1 = 0.5 (rising)
        assert basis_mid[1, 0].item() == pytest.approx(0.5)
        assert basis_mid[1, 1].item() == pytest.approx(0.5)

    def test_partition_of_unity(self):
        """Test that sum of all basis functions equals 1 at interior points.

        This is a fundamental property of B-splines: they form a partition of unity.
        For open (uniform) knot vectors, partition of unity holds only in the
        interior [t_degree, t_{n_knots-degree-1}].
        For clamped knot vectors (repeated boundary knots), it holds over [a, b].
        """
        from torchscience.spline import b_spline_basis

        # Test with clamped knot vectors (partition of unity over full domain)
        for degree in [0, 1, 2, 3]:
            # Create clamped knot vector: degree+1 repeated knots at each end
            # with some interior knots
            n_interior = 3
            interior_knots = torch.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            boundary_left = torch.zeros(degree + 1, dtype=torch.float64)
            boundary_right = torch.ones(degree + 1, dtype=torch.float64)
            knots = torch.cat([boundary_left, interior_knots, boundary_right])

            # Test at many points across the full domain
            t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)

            basis = b_spline_basis(t, knots, degree)

            # Sum over basis functions should be 1
            basis_sum = basis.sum(dim=-1)
            expected = torch.ones_like(basis_sum)

            torch.testing.assert_close(
                basis_sum,
                expected,
                atol=1e-10,
                rtol=1e-10,
                msg=f"Partition of unity failed for degree {degree}",
            )

    def test_local_support(self):
        """Test that basis i is zero outside its local support [t_i, t_{i+degree+1}]."""
        from torchscience.spline import b_spline_basis

        # Uniform knots: [0, 1, 2, 3, 4, 5, 6, 7]
        knots = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float64
        )
        degree = 2

        # n_basis = 8 - 2 - 1 = 5
        # Basis 0 has support [0, 3]
        # Basis 1 has support [1, 4]
        # etc.

        # Test basis function 1 (i=1): support [1, 4]
        # Should be zero at t < 1 and t > 4

        # Points outside support of basis 1
        t_outside = torch.tensor([0.5, 4.5, 5.5], dtype=torch.float64)
        basis_outside = b_spline_basis(t_outside, knots, degree, i=1)
        expected_zeros = torch.zeros(3, dtype=torch.float64)
        torch.testing.assert_close(
            basis_outside, expected_zeros, atol=1e-12, rtol=1e-12
        )

        # Points inside support of basis 1
        t_inside = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)
        basis_inside = b_spline_basis(t_inside, knots, degree, i=1)
        # Should be positive (non-zero)
        assert torch.all(basis_inside > 0)

    def test_specific_basis_function(self):
        """Test that the i parameter returns only the i-th basis function."""
        from torchscience.spline import b_spline_basis

        knots = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )
        degree = 2

        t = torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64)

        # Get all basis functions
        all_basis = b_spline_basis(t, knots, degree)  # shape (3, n_basis)

        # n_basis = 6 - 2 - 1 = 3
        assert all_basis.shape == (3, 3)

        # Get specific basis functions and compare
        for i in range(3):
            single_basis = b_spline_basis(t, knots, degree, i=i)
            # Shape should be (3,) - the query shape
            assert single_basis.shape == (3,)
            torch.testing.assert_close(single_basis, all_basis[:, i])

    def test_scipy_comparison(self):
        """Compare with scipy.interpolate.BSpline.basis_element."""
        scipy = pytest.importorskip("scipy")
        from scipy.interpolate import BSpline

        from torchscience.spline import b_spline_basis

        # Uniform knots
        knots = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64
        )
        degree = 3

        # n_basis = 7 - 3 - 1 = 3
        # Test in the interior of each basis function's support
        # Basis 0 has support [0, 4], Basis 1 has support [1, 5], Basis 2 has support [2, 6]
        # The intersection where all are non-zero is [2, 4]
        # But we want to test each individually within its support
        t = torch.linspace(0.5, 5.5, 50, dtype=torch.float64)

        # Get our basis functions
        our_basis = b_spline_basis(t, knots, degree)

        # Compare with scipy for each basis function
        for i in range(3):
            # scipy.interpolate.BSpline.basis_element uses knots[i:i+k+2]
            # for k=degree and basis function i
            local_knots = knots[i : i + degree + 2].numpy()
            scipy_basis_elem = BSpline.basis_element(local_knots)

            # Only compare within the support of this basis function
            support_left = knots[i].item()
            support_right = knots[i + degree + 1].item()

            # Find points within support (with small epsilon to avoid boundary issues)
            eps = 1e-10
            in_support = (t >= support_left + eps) & (t <= support_right - eps)

            if in_support.any():
                t_in_support = t[in_support]

                # Evaluate scipy basis function
                scipy_vals = scipy_basis_elem(t_in_support.numpy())
                torch_vals = our_basis[in_support, i].numpy()

                torch.testing.assert_close(
                    torch.tensor(torch_vals, dtype=torch.float64),
                    torch.tensor(scipy_vals, dtype=torch.float64),
                    atol=1e-10,
                    rtol=1e-10,
                )

    def test_gradcheck(self):
        """Verify that gradients work correctly for the basis functions."""
        from torch.autograd import gradcheck

        from torchscience.spline import b_spline_basis

        knots = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64
        )
        degree = 3

        # Query points that require gradients
        # Keep away from knots where gradient may be discontinuous
        t = torch.tensor(
            [1.25, 2.5, 3.75], dtype=torch.float64, requires_grad=True
        )

        def basis_fn(x):
            return b_spline_basis(x, knots, degree)

        assert gradcheck(basis_fn, (t,), eps=1e-6, atol=1e-4)

    def test_degree_validation(self):
        """Test that invalid degree raises DegreeError."""
        from torchscience.spline import DegreeError, b_spline_basis

        knots = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

        # Negative degree
        with pytest.raises(DegreeError):
            b_spline_basis(torch.tensor([0.5]), knots, degree=-1)

        # Degree too high for knot count (need n_knots >= degree + 2)
        # 4 knots, degree 3: need 4 >= 5, which is false
        with pytest.raises(DegreeError):
            b_spline_basis(torch.tensor([0.5]), knots, degree=3)

    def test_knot_validation(self):
        """Test that non-monotonic knots raise KnotError."""
        from torchscience.spline import KnotError, b_spline_basis

        # Non-monotonic knots
        knots = torch.tensor([0.0, 2.0, 1.0, 3.0], dtype=torch.float64)

        with pytest.raises(KnotError):
            b_spline_basis(torch.tensor([0.5]), knots, degree=1)

    def test_rightmost_point_included(self):
        """Test that t equals the last knot is handled correctly.

        For clamped B-splines, the rightmost point should have the last
        basis function equal to 1.
        """
        from torchscience.spline import b_spline_basis

        # Use clamped knot vector for degree 1
        knots = torch.tensor(
            [0.0, 0.0, 1.0, 2.0, 3.0, 3.0], dtype=torch.float64
        )
        degree = 1

        # n_basis = 6 - 1 - 1 = 4
        # Test at exactly the last knot
        t = torch.tensor([3.0], dtype=torch.float64)
        basis = b_spline_basis(t, knots, degree)

        # For clamped splines, partition of unity should hold
        assert basis.sum().item() == pytest.approx(1.0)

        # The last basis function should be 1 at the right boundary
        assert basis[0, -1].item() == pytest.approx(1.0)

    def test_repeated_knots(self):
        """Test handling of repeated (multiplicity > 1) knots."""
        from torchscience.spline import b_spline_basis

        # Clamped knot vector for degree 2: triple knots at ends
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        degree = 2

        # n_basis = 7 - 2 - 1 = 4
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        basis = b_spline_basis(t, knots, degree)

        # Should still satisfy partition of unity
        basis_sum = basis.sum(dim=-1)
        expected = torch.ones(5, dtype=torch.float64)
        torch.testing.assert_close(basis_sum, expected, atol=1e-10, rtol=1e-10)

        # At t=0, first basis should be 1 (clamped endpoint)
        assert basis[0, 0].item() == pytest.approx(1.0)

        # At t=1, last basis should be 1 (clamped endpoint)
        assert basis[4, -1].item() == pytest.approx(1.0)

    def test_batch_query_shape(self):
        """Test that arbitrary query shapes are preserved."""
        from torchscience.spline import b_spline_basis

        # Use clamped knot vector so partition of unity holds everywhere
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float64
        )
        degree = 2

        # n_basis = 8 - 2 - 1 = 5

        # 2D query shape
        t = torch.linspace(0.5, 2.5, 12, dtype=torch.float64).reshape(3, 4)
        basis = b_spline_basis(t, knots, degree)

        # Output shape should be (3, 4, 5)
        assert basis.shape == (3, 4, 5)

        # Partition of unity should hold for clamped splines
        basis_sum = basis.sum(dim=-1)
        expected = torch.ones(3, 4, dtype=torch.float64)
        torch.testing.assert_close(basis_sum, expected, atol=1e-10, rtol=1e-10)

    def test_scalar_query(self):
        """Test that scalar (0-d tensor) query works correctly."""
        from torchscience.spline import b_spline_basis

        # Use clamped knot vector so partition of unity holds
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=torch.float64
        )
        degree = 2

        # Scalar query
        t = torch.tensor(1.5, dtype=torch.float64)
        basis = b_spline_basis(t, knots, degree)

        # n_basis = 8 - 2 - 1 = 5
        # For scalar input, output shape should be (n_basis,)
        assert basis.shape == (5,)

        # Partition of unity
        assert basis.sum().item() == pytest.approx(1.0)


class TestBSplineFit:
    """Tests for b_spline_fit function."""

    def test_fit_exact_polynomial(self):
        """Fitting a polynomial of degree d with degree d spline should be exact.

        A B-spline of degree d can exactly represent any polynomial of degree <= d.
        """
        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create data from a quadratic polynomial: y = x^2
        x = torch.linspace(0.0, 2.0, 20, dtype=torch.float64)
        y = x**2

        # Fit with degree 2 (quadratic) spline
        spline = b_spline_fit(x, y, degree=2, n_knots=5)

        # Evaluate at the original points - should be exact
        y_fit = b_spline_evaluate(spline, x)
        torch.testing.assert_close(y_fit, y, atol=1e-10, rtol=1e-10)

        # Also evaluate at intermediate points
        x_test = torch.linspace(0.0, 2.0, 50, dtype=torch.float64)
        y_expected = x_test**2
        y_test = b_spline_evaluate(spline, x_test)
        torch.testing.assert_close(y_test, y_expected, atol=1e-8, rtol=1e-8)

    def test_fit_sine(self):
        """Fitting sin(x) should give good approximation."""
        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create data from sin(x)
        x = torch.linspace(0.0, 2 * torch.pi, 50, dtype=torch.float64)
        y = torch.sin(x)

        # Fit with cubic spline
        spline = b_spline_fit(x, y, degree=3, n_knots=10)

        # Evaluate at test points
        x_test = torch.linspace(0.0, 2 * torch.pi, 100, dtype=torch.float64)
        y_expected = torch.sin(x_test)
        y_fit = b_spline_evaluate(spline, x_test)

        # Should be a good approximation
        torch.testing.assert_close(y_fit, y_expected, atol=1e-3, rtol=1e-3)

    def test_fit_with_explicit_knots(self):
        """Can provide custom knot vector."""
        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create data from a simple function
        x = torch.linspace(0.0, 1.0, 30, dtype=torch.float64)
        y = x * (1 - x)  # Parabola with peak at 0.5

        # Provide explicit clamped knot vector for degree 3
        # n_control = n_knots - degree - 1
        # With 12 knots and degree 3: n_control = 12 - 3 - 1 = 8
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        )

        spline = b_spline_fit(x, y, degree=3, knots=knots)

        # Should be able to evaluate
        y_fit = b_spline_evaluate(spline, x)

        # Should be a good approximation
        torch.testing.assert_close(y_fit, y, atol=1e-4, rtol=1e-4)

    def test_fit_multidimensional(self):
        """Fitting multi-dimensional y values works."""
        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create 2D curve data: circle in 2D
        t = torch.linspace(0.0, 2 * torch.pi, 30, dtype=torch.float64)
        x = t  # Parameter values
        y = torch.stack([torch.cos(t), torch.sin(t)], dim=-1)  # (30, 2)

        # Fit with cubic spline
        spline = b_spline_fit(x, y, degree=3, n_knots=8)

        # Evaluate at original points
        y_fit = b_spline_evaluate(spline, x)

        # Output should have correct shape
        assert y_fit.shape == (30, 2)

        # Should be a good approximation
        torch.testing.assert_close(y_fit, y, atol=1e-3, rtol=1e-3)

    def test_fit_interpolates_endpoints(self):
        """With clamped knots, spline passes near endpoints."""
        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create some data
        x = torch.linspace(0.0, 1.0, 20, dtype=torch.float64)
        y = torch.sin(torch.pi * x)

        # Fit with cubic spline (uses clamped knots by default)
        spline = b_spline_fit(x, y, degree=3, n_knots=5)

        # Evaluate at endpoints
        y_at_0 = b_spline_evaluate(
            spline, torch.tensor(0.0, dtype=torch.float64)
        )
        y_at_1 = b_spline_evaluate(
            spline, torch.tensor(1.0, dtype=torch.float64)
        )

        # Should be close to the data endpoints (y[0] = 0, y[-1] = 0)
        assert y_at_0.item() == pytest.approx(y[0].item(), abs=1e-2)
        assert y_at_1.item() == pytest.approx(y[-1].item(), abs=1e-2)

    def test_scipy_comparison(self):
        """Compare with scipy.interpolate.make_lsq_spline."""
        scipy = pytest.importorskip("scipy")
        from scipy.interpolate import make_lsq_spline

        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create data
        x_np = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y_np = [0.0, 0.31, 0.59, 0.81, 0.95, 1.0, 0.95, 0.81, 0.59, 0.31, 0.0]

        x = torch.tensor(x_np, dtype=torch.float64)
        y = torch.tensor(y_np, dtype=torch.float64)

        # Clamped knot vector for degree 3 with 3 interior knots
        degree = 3
        knots_np = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
        knots = torch.tensor(knots_np, dtype=torch.float64)

        # Fit with scipy
        import numpy as np

        scipy_spline = make_lsq_spline(
            np.array(x_np), np.array(y_np), knots_np, k=degree
        )

        # Fit with torchscience
        torch_spline = b_spline_fit(x, y, degree=degree, knots=knots)

        # Compare evaluations
        x_test = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        y_scipy = scipy_spline(x_test.numpy())
        y_torch = b_spline_evaluate(torch_spline, x_test)

        torch.testing.assert_close(
            y_torch,
            torch.tensor(y_scipy, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_gradcheck(self):
        """Gradients flow through fitting."""
        from torch.autograd import gradcheck

        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create data with requires_grad
        x = torch.linspace(0.0, 1.0, 15, dtype=torch.float64)
        y = torch.tensor(
            [
                0.0,
                0.2,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.85,
                0.9,
                0.92,
                0.94,
                0.95,
                0.96,
                0.97,
                1.0,
            ],
            dtype=torch.float64,
            requires_grad=True,
        )

        # Define function that fits and evaluates
        def fit_and_evaluate(y_data):
            spline = b_spline_fit(x, y_data, degree=3, n_knots=3)
            # Evaluate at a few points (avoid knots)
            t = torch.tensor([0.15, 0.45, 0.85], dtype=torch.float64)
            return b_spline_evaluate(spline, t)

        assert gradcheck(fit_and_evaluate, (y,), eps=1e-6, atol=1e-4)

    def test_default_knot_count(self):
        """Default knot count is computed based on data size."""
        from torchscience.spline import b_spline_fit

        # Create data
        x = torch.linspace(0.0, 1.0, 30, dtype=torch.float64)
        y = x**2

        # Fit without specifying n_knots or knots
        spline = b_spline_fit(x, y, degree=3)

        # Should have created a valid spline
        # n_knots = n_interior + 2*(degree+1)
        # n_control = n_knots - degree - 1
        n_knots = spline.knots.shape[0]
        n_control = spline.control_points.shape[0]

        assert n_control == n_knots - 3 - 1  # degree = 3
        assert n_knots > 2 * (3 + 1)  # At least boundary knots

    def test_extrapolate_modes(self):
        """Different extrapolation modes work."""
        from torchscience.spline import (
            ExtrapolationError,
            b_spline_evaluate,
            b_spline_fit,
        )

        x = torch.linspace(0.0, 1.0, 20, dtype=torch.float64)
        y = x**2

        # Test error mode
        spline_error = b_spline_fit(
            x, y, degree=3, n_knots=3, extrapolate="error"
        )
        with pytest.raises(ExtrapolationError):
            b_spline_evaluate(
                spline_error, torch.tensor([-0.1], dtype=torch.float64)
            )

        # Test clamp mode
        spline_clamp = b_spline_fit(
            x, y, degree=3, n_knots=3, extrapolate="clamp"
        )
        y_clamp = b_spline_evaluate(
            spline_clamp, torch.tensor([-0.1, 1.1], dtype=torch.float64)
        )
        y_at_0 = b_spline_evaluate(
            spline_clamp, torch.tensor([0.0], dtype=torch.float64)
        )
        y_at_1 = b_spline_evaluate(
            spline_clamp, torch.tensor([1.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            y_clamp[0], y_at_0.squeeze(), atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            y_clamp[1], y_at_1.squeeze(), atol=1e-10, rtol=1e-10
        )


class TestBSplineDerivative:
    """Tests for b_spline_derivative function."""

    def test_derivative_of_linear(self):
        """Derivative of linear spline is constant.

        A linear B-spline (degree 1) has piecewise constant derivative.
        For a linear spline that interpolates y = 2*x on [0, 1],
        the derivative should be 2 everywhere.
        """
        from torchscience.spline import (
            BSpline,
            b_spline_derivative,
            b_spline_evaluate,
        )

        # Degree 1 (linear) B-spline with clamped knots
        # n_knots = n_control + degree + 1 = 2 + 1 + 1 = 4
        # For y = 2*x on [0, 1]: control points at [0, 2]
        knots = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64)
        control_points = torch.tensor([0.0, 2.0], dtype=torch.float64)

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=1,
            extrapolate="error",
            batch_size=[],
        )

        # Compute derivative
        derivative = b_spline_derivative(spline)

        # The derivative of a degree 1 spline is a degree 0 spline (constant)
        assert derivative.degree == 0

        # Evaluate derivative at multiple points
        t = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
        dy = b_spline_evaluate(derivative, t)

        # Derivative of y = 2*x should be 2
        expected = torch.full((10,), 2.0, dtype=torch.float64)
        torch.testing.assert_close(dy, expected, atol=1e-10, rtol=1e-10)

    def test_derivative_of_quadratic(self):
        """Derivative of quadratic spline is linear.

        A quadratic B-spline (degree 2) has piecewise linear derivative.
        For a quadratic spline representing x^2, the derivative should be 2*x.
        """
        from torchscience.spline import (
            b_spline_derivative,
            b_spline_evaluate,
            b_spline_fit,
        )

        # Fit a quadratic B-spline to x^2 data
        x = torch.linspace(0.0, 2.0, 20, dtype=torch.float64)
        y = x**2

        spline = b_spline_fit(x, y, degree=2, n_knots=5)

        # Compute derivative
        derivative = b_spline_derivative(spline)

        # The derivative of a degree 2 spline is a degree 1 spline (linear)
        assert derivative.degree == 1

        # Evaluate derivative at interior points
        t = torch.linspace(0.1, 1.9, 15, dtype=torch.float64)
        dy = b_spline_evaluate(derivative, t)

        # Derivative of x^2 should be approximately 2*x
        expected = 2 * t
        torch.testing.assert_close(dy, expected, atol=1e-2, rtol=1e-2)

    def test_derivative_of_cubic(self):
        """Derivative of x^3 is 3x^2.

        A cubic B-spline (degree 3) representing x^3 should have derivative 3*x^2.
        """
        from torchscience.spline import (
            b_spline_derivative,
            b_spline_evaluate,
            b_spline_fit,
        )

        # Fit a cubic B-spline to x^3 data
        x = torch.linspace(0.0, 2.0, 30, dtype=torch.float64)
        y = x**3

        spline = b_spline_fit(x, y, degree=3, n_knots=8)

        # Compute derivative
        derivative = b_spline_derivative(spline)

        # The derivative of a degree 3 spline is a degree 2 spline (quadratic)
        assert derivative.degree == 2

        # Evaluate derivative at interior points
        t = torch.linspace(0.1, 1.9, 20, dtype=torch.float64)
        dy = b_spline_evaluate(derivative, t)

        # Derivative of x^3 should be approximately 3*x^2
        expected = 3 * t**2
        torch.testing.assert_close(dy, expected, atol=5e-2, rtol=5e-2)

    def test_second_derivative(self):
        """Second derivative of cubic is linear.

        The second derivative of x^3 is 6*x.
        """
        from torchscience.spline import (
            b_spline_derivative,
            b_spline_evaluate,
            b_spline_fit,
        )

        # Fit a cubic B-spline to x^3 data
        x = torch.linspace(0.0, 2.0, 30, dtype=torch.float64)
        y = x**3

        spline = b_spline_fit(x, y, degree=3, n_knots=8)

        # Compute second derivative
        second_derivative = b_spline_derivative(spline, order=2)

        # The second derivative of a degree 3 spline is a degree 1 spline (linear)
        assert second_derivative.degree == 1

        # Evaluate second derivative at interior points
        t = torch.linspace(0.2, 1.8, 15, dtype=torch.float64)
        d2y = b_spline_evaluate(second_derivative, t)

        # Second derivative of x^3 should be approximately 6*x
        expected = 6 * t
        torch.testing.assert_close(d2y, expected, atol=1e-1, rtol=1e-1)

    def test_derivative_preserves_domain(self):
        """Derivative has same valid domain as original spline.

        The derivative spline should be evaluable over the same domain
        as the original spline.
        """
        from torchscience.spline import (
            BSpline,
            b_spline_derivative,
            b_spline_evaluate,
        )

        # Clamped cubic B-spline on [0, 1]
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [0.0, 1.0, 2.0, 1.0, 0.0], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Compute derivative
        derivative = b_spline_derivative(spline)

        # Should be able to evaluate at the original domain endpoints
        t_endpoints = torch.tensor([0.0, 1.0], dtype=torch.float64)
        dy = b_spline_evaluate(derivative, t_endpoints)

        # Should not raise and should return valid values
        assert dy.shape == (2,)
        assert torch.isfinite(dy).all()

        # Should also evaluate at interior points
        t_interior = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        dy_interior = b_spline_evaluate(derivative, t_interior)
        assert dy_interior.shape == (50,)
        assert torch.isfinite(dy_interior).all()

    def test_scipy_comparison(self):
        """Compare with scipy BSpline derivative."""
        scipy = pytest.importorskip("scipy")
        from scipy.interpolate import BSpline as ScipyBSpline

        from torchscience.spline import (
            BSpline,
            b_spline_derivative,
            b_spline_evaluate,
        )

        # Clamped cubic B-spline
        knots_np = [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0]
        control_np = [0.0, 0.5, 1.0, 0.8, 0.3, 0.1, 0.5]
        degree = 3

        knots = torch.tensor(knots_np, dtype=torch.float64)
        control_points = torch.tensor(control_np, dtype=torch.float64)

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=degree,
            extrapolate="error",
            batch_size=[],
        )

        # Create scipy equivalent and compute derivative
        scipy_spline = ScipyBSpline(knots_np, control_np, degree)
        scipy_derivative = scipy_spline.derivative()

        # Compute our derivative
        torch_derivative = b_spline_derivative(spline)

        # Evaluate at many points in the interior
        t = torch.linspace(0.01, 0.99, 50, dtype=torch.float64)
        y_torch = b_spline_evaluate(torch_derivative, t)
        y_scipy = scipy_derivative(t.numpy())

        torch.testing.assert_close(
            y_torch,
            torch.tensor(y_scipy, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )


class TestBSplineConvenience:
    """Tests for b_spline convenience function."""

    def test_b_spline_convenience(self):
        """Test basic usage of b_spline convenience function."""
        from torchscience.spline import b_spline

        # Create data from sin(x)
        x = torch.linspace(0, 2 * torch.pi, 30, dtype=torch.float64)
        y = torch.sin(x)

        # Create callable using convenience function
        f = b_spline(x, y, degree=3, n_knots=8)

        # Evaluate at original points - should approximately match original y values
        y_eval = f(x)
        torch.testing.assert_close(y_eval, y, atol=1e-3, rtol=1e-3)

        # Evaluate at intermediate points
        x_mid = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
        y_mid = f(x_mid)

        # Results should be reasonable (close to sin values)
        y_expected = torch.sin(x_mid)
        torch.testing.assert_close(y_mid, y_expected, atol=1e-2, rtol=1e-2)

    def test_b_spline_convenience_default_degree(self):
        """Test that default degree=3 works correctly."""
        from torchscience.spline import b_spline

        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = x**2

        # Create without specifying degree (default is 3)
        f = b_spline(x, y, n_knots=5)

        # Should work and produce reasonable results
        result = f(torch.tensor([0.5], dtype=torch.float64))
        expected = torch.tensor([0.25], dtype=torch.float64)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_b_spline_convenience_different_degrees(self):
        """Test b_spline with different degrees."""
        from torchscience.spline import b_spline

        x = torch.linspace(0, 1, 30, dtype=torch.float64)
        y = x**2

        # Test degrees 1, 2, 3
        for degree in [1, 2, 3]:
            f = b_spline(x, y, degree=degree, n_knots=5)
            result = f(torch.tensor([0.5], dtype=torch.float64))
            assert result.shape == (1,)
            # Quadratic function should be well-approximated by degree >= 2
            if degree >= 2:
                expected = torch.tensor([0.25], dtype=torch.float64)
                torch.testing.assert_close(
                    result, expected, atol=5e-2, rtol=5e-2
                )

    def test_b_spline_convenience_extrapolate_error(self):
        """Test that extrapolate='error' raises ExtrapolationError."""
        from torchscience.spline import ExtrapolationError, b_spline

        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = x**2

        # Default extrapolate='error' should raise on out-of-domain query
        f = b_spline(x, y, n_knots=5, extrapolate="error")

        # Query below domain
        with pytest.raises(ExtrapolationError):
            f(torch.tensor([-0.1], dtype=torch.float64))

        # Query above domain
        with pytest.raises(ExtrapolationError):
            f(torch.tensor([1.1], dtype=torch.float64))

    def test_b_spline_convenience_extrapolate_clamp(self):
        """Test that extrapolate='clamp' clamps to boundary values."""
        from torchscience.spline import b_spline

        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = torch.linspace(0, 2, 20, dtype=torch.float64)  # y = 2*x

        f = b_spline(x, y, n_knots=5, extrapolate="clamp")

        # Query outside domain should return boundary values
        y_outside = f(torch.tensor([-1.0, 2.0], dtype=torch.float64))
        y_at_0 = f(torch.tensor([0.0], dtype=torch.float64))
        y_at_1 = f(torch.tensor([1.0], dtype=torch.float64))

        torch.testing.assert_close(
            y_outside[0], y_at_0.squeeze(), atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            y_outside[1], y_at_1.squeeze(), atol=1e-10, rtol=1e-10
        )

    def test_b_spline_convenience_returns_callable(self):
        """Test that b_spline returns a callable."""
        from torchscience.spline import b_spline

        x = torch.linspace(0, 1, 20, dtype=torch.float64)
        y = x**2

        f = b_spline(x, y, n_knots=5)

        # Should be callable
        assert callable(f)

        # Should accept tensor input
        result = f(torch.tensor([0.5], dtype=torch.float64))
        assert isinstance(result, torch.Tensor)

    def test_b_spline_convenience_multidimensional(self):
        """Test b_spline with multi-dimensional y values."""
        from torchscience.spline import b_spline

        # 2D curve: circle in 2D
        t = torch.linspace(0, 2 * torch.pi, 30, dtype=torch.float64)
        x = t  # Parameter values
        y = torch.stack([torch.cos(t), torch.sin(t)], dim=-1)  # (30, 2)

        f = b_spline(x, y, degree=3, n_knots=8)

        # Evaluate at original points
        y_eval = f(x)
        assert y_eval.shape == (30, 2)

        # Should be a good approximation
        torch.testing.assert_close(y_eval, y, atol=1e-3, rtol=1e-3)

        # Evaluate at single point
        y_mid = f(torch.tensor([torch.pi], dtype=torch.float64))
        assert y_mid.shape == (1, 2)

    def test_b_spline_convenience_default_n_knots(self):
        """Test that n_knots=None uses a reasonable default."""
        from torchscience.spline import b_spline

        x = torch.linspace(0, 1, 50, dtype=torch.float64)
        y = torch.sin(x * torch.pi)

        # Create without specifying n_knots
        f = b_spline(x, y, degree=3)

        # Should work and produce reasonable results
        y_eval = f(x)
        torch.testing.assert_close(y_eval, y, atol=1e-2, rtol=1e-2)


class TestBSplineBatching:
    """Tests for batched B-spline operations."""

    def test_basis_batched_query(self):
        """Test basis evaluation with batched query points (2D query shape)."""
        from torchscience.spline import b_spline_basis

        # Clamped cubic knot vector
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        degree = 3

        # n_basis = 9 - 3 - 1 = 5

        # Query with shape (3, 4) - 3 batches, 4 queries each
        t = torch.linspace(0.1, 0.9, 12, dtype=torch.float64).reshape(3, 4)
        basis = b_spline_basis(t, knots, degree)

        # Output shape should be (3, 4, 5)
        assert basis.shape == (3, 4, 5)

        # Verify partition of unity (sum of basis functions = 1)
        basis_sum = basis.sum(dim=-1)
        expected = torch.ones(3, 4, dtype=torch.float64)
        torch.testing.assert_close(basis_sum, expected, atol=1e-10, rtol=1e-10)

        # Verify values match flat evaluation
        t_flat = t.flatten()
        basis_flat = b_spline_basis(t_flat, knots, degree)
        torch.testing.assert_close(
            basis, basis_flat.reshape(3, 4, 5), atol=1e-10, rtol=1e-10
        )

    def test_fit_batched_y(self):
        """Test fitting with batched y values (multiple curves)."""
        from torchscience.spline import b_spline_evaluate, b_spline_fit

        # Create data for 3 curves
        x = torch.linspace(0.0, 2 * torch.pi, 30, dtype=torch.float64)
        # y has shape (30, 3) - 3 curves
        y = torch.stack(
            [
                torch.sin(x),
                torch.cos(x),
                x / (2 * torch.pi),  # Linear 0 to 1
            ],
            dim=-1,
        )

        spline = b_spline_fit(x, y, degree=3, n_knots=8)

        # Control points should have shape (n_control, 3)
        n_knots = spline.knots.shape[0]
        n_control = n_knots - 3 - 1
        assert spline.control_points.shape == (n_control, 3)

        # Evaluate at test points
        t = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64)
        y_eval = b_spline_evaluate(spline, t)

        # Should have shape (4, 3) - 4 query points, 3 curves
        assert y_eval.shape == (4, 3)

        # Verify each curve matches individual fit
        for i in range(3):
            y_col = y[:, i]
            spline_single = b_spline_fit(x, y_col, degree=3, n_knots=8)
            y_single = b_spline_evaluate(spline_single, t)
            torch.testing.assert_close(
                y_eval[:, i], y_single, atol=1e-8, rtol=1e-8
            )

    def test_evaluate_batched_query(self):
        """Test evaluation with batched query points (2D query shape)."""
        from torchscience.spline import BSpline, b_spline_evaluate

        # Create a simple cubic B-spline
        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        control_points = torch.tensor(
            [0.0, 1.0, 2.0, 1.0, 0.5], dtype=torch.float64
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Query with shape (batch, n_queries) = (4, 5)
        t = torch.linspace(0.1, 0.9, 20, dtype=torch.float64).reshape(4, 5)
        y_eval = b_spline_evaluate(spline, t)

        # Output shape should match query shape
        assert y_eval.shape == (4, 5)

        # Verify values match flat evaluation
        t_flat = t.flatten()
        y_flat = b_spline_evaluate(spline, t_flat)
        torch.testing.assert_close(
            y_eval, y_flat.reshape(4, 5), atol=1e-10, rtol=1e-10
        )

    def test_derivative_batched(self):
        """Test derivative of batched B-spline (multiple curves)."""
        from torchscience.spline import (
            b_spline_derivative,
            b_spline_evaluate,
            b_spline_fit,
        )

        # Fit two curves: x^2 and x^3
        x = torch.linspace(0.0, 2.0, 30, dtype=torch.float64)
        y = torch.stack([x**2, x**3], dim=-1)

        spline = b_spline_fit(x, y, degree=3, n_knots=8)

        # Get derivative
        deriv_spline = b_spline_derivative(spline, order=1)

        # Evaluate at interior test points
        t = torch.linspace(0.2, 1.8, 10, dtype=torch.float64)
        deriv_eval = b_spline_evaluate(deriv_spline, t)

        # Should have shape (10, 2) - 10 query points, 2 curves
        assert deriv_eval.shape == (10, 2)

        # Expected: derivative of x^2 is 2x, derivative of x^3 is 3x^2
        expected_col0 = 2 * t  # d/dx(x^2) = 2x
        expected_col1 = 3 * t**2  # d/dx(x^3) = 3x^2

        # Use looser tolerance since B-spline is an approximation
        torch.testing.assert_close(
            deriv_eval[:, 0], expected_col0, atol=5e-2, rtol=5e-2
        )
        torch.testing.assert_close(
            deriv_eval[:, 1], expected_col1, atol=5e-2, rtol=5e-2
        )

    def test_gradcheck_batched(self):
        """Test that gradients flow correctly through batched B-spline operations."""
        from torch.autograd import gradcheck

        from torchscience.spline import BSpline, b_spline_evaluate

        knots = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64
        )
        # Multi-dimensional control points (5, 2) - 2 curves
        control_points = torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [2.0, 1.5], [1.0, 0.5], [0.5, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )

        spline = BSpline(
            knots=knots,
            control_points=control_points,
            degree=3,
            extrapolate="error",
            batch_size=[],
        )

        # Query points (avoid knots where gradient may be discontinuous)
        t = torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64)

        def eval_fn(cp):
            s = BSpline(
                knots=knots,
                control_points=cp,
                degree=3,
                extrapolate="error",
                batch_size=[],
            )
            return b_spline_evaluate(s, t)

        assert gradcheck(eval_fn, (control_points,), eps=1e-6, atol=1e-4)
