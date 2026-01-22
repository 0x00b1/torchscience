"""Tests for Bezier curve functions."""

import pytest
import torch

from torchscience.spline import (
    BezierCurve,
    ExtrapolationError,
    bezier,
    bezier_derivative,
    bezier_derivative_evaluate,
    bezier_evaluate,
    bezier_split,
)


class TestBezierCurve:
    """Tests for BezierCurve tensorclass."""

    def test_degree_property(self):
        """Should return correct degree."""
        # Linear (degree 1)
        cp = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=cp, extrapolate="error", batch_size=[]
        )
        assert curve.degree == 1

        # Quadratic (degree 2)
        cp = torch.tensor([[0.0], [0.5], [1.0]])
        curve = BezierCurve(
            control_points=cp, extrapolate="error", batch_size=[]
        )
        assert curve.degree == 2

        # Cubic (degree 3)
        cp = torch.tensor([[0.0], [0.25], [0.75], [1.0]])
        curve = BezierCurve(
            control_points=cp, extrapolate="error", batch_size=[]
        )
        assert curve.degree == 3


class TestBezierEvaluate:
    """Tests for bezier_evaluate function."""

    def test_linear_bezier(self):
        """Linear Bezier should be linear interpolation."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        result = bezier_evaluate(curve, t)

        expected = t.unsqueeze(-1)
        assert torch.allclose(result, expected)

    def test_quadratic_bezier_2d(self):
        """Quadratic Bezier in 2D."""
        # Parabola-like curve
        control_points = torch.tensor([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        # At endpoints
        assert torch.allclose(
            bezier_evaluate(curve, torch.tensor(0.0)),
            torch.tensor([0.0, 0.0]),
        )
        assert torch.allclose(
            bezier_evaluate(curve, torch.tensor(1.0)),
            torch.tensor([1.0, 0.0]),
        )

        # At midpoint: B(0.5) = 0.25*P0 + 0.5*P1 + 0.25*P2 = [0.5, 0.5]
        result = bezier_evaluate(curve, torch.tensor(0.5))
        assert torch.allclose(result, torch.tensor([0.5, 0.5]))

    def test_cubic_bezier(self):
        """Cubic Bezier curve."""
        control_points = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
        )
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        # Endpoints
        assert torch.allclose(
            bezier_evaluate(curve, torch.tensor(0.0)),
            torch.tensor([0.0, 0.0]),
        )
        assert torch.allclose(
            bezier_evaluate(curve, torch.tensor(1.0)),
            torch.tensor([1.0, 0.0]),
        )

    def test_evaluate_scalar_query(self):
        """Should handle scalar parameter."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        result = bezier_evaluate(curve, torch.tensor(0.5))
        assert result.shape == (1,)
        assert torch.allclose(result, torch.tensor([0.5]))

    def test_evaluate_batch_query(self):
        """Should handle batched parameters."""
        control_points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        t = torch.tensor([[0.0, 0.5], [0.75, 1.0]])  # Shape (2, 2)
        result = bezier_evaluate(curve, t)

        assert result.shape == (2, 2, 2)  # (2, 2, value_dim)

    def test_extrapolate_error(self):
        """Should raise error for out-of-range parameters."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        with pytest.raises(ExtrapolationError):
            bezier_evaluate(curve, torch.tensor(-0.1))

        with pytest.raises(ExtrapolationError):
            bezier_evaluate(curve, torch.tensor(1.1))

    def test_extrapolate_clamp(self):
        """Should clamp parameters to [0, 1]."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="clamp", batch_size=[]
        )

        result_neg = bezier_evaluate(curve, torch.tensor(-0.5))
        result_pos = bezier_evaluate(curve, torch.tensor(1.5))

        assert torch.allclose(result_neg, torch.tensor([0.0]))
        assert torch.allclose(result_pos, torch.tensor([1.0]))

    def test_extrapolate_extend(self):
        """Should allow extrapolation."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points,
            extrapolate="extrapolate",
            batch_size=[],
        )

        # Linear extrapolation should work
        result_neg = bezier_evaluate(curve, torch.tensor(-0.5))
        result_pos = bezier_evaluate(curve, torch.tensor(1.5))

        assert torch.allclose(result_neg, torch.tensor([-0.5]))
        assert torch.allclose(result_pos, torch.tensor([1.5]))

    def test_gradcheck(self):
        """Gradient check for Bezier evaluation."""
        control_points = torch.tensor(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]], dtype=torch.float64
        )
        curve = BezierCurve(
            control_points=control_points, extrapolate="clamp", batch_size=[]
        )

        t = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )

        # Just verify gradients exist and are reasonable
        result = bezier_evaluate(curve, t)
        loss = result.sum()
        loss.backward()

        assert t.grad is not None


class TestBezierDerivative:
    """Tests for bezier_derivative function."""

    def test_linear_derivative_is_constant(self):
        """Derivative of linear Bezier is constant."""
        control_points = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        deriv = bezier_derivative(curve)

        assert deriv.degree == 0
        assert deriv.control_points.shape == (1, 2)
        assert torch.allclose(deriv.control_points, torch.tensor([[1.0, 2.0]]))

    def test_quadratic_derivative_is_linear(self):
        """Derivative of quadratic Bezier is linear."""
        control_points = torch.tensor([[0.0], [1.0], [0.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        deriv = bezier_derivative(curve)

        assert deriv.degree == 1
        assert deriv.control_points.shape == (2, 1)

        # Q_0 = 2 * (P_1 - P_0) = 2 * (1 - 0) = 2
        # Q_1 = 2 * (P_2 - P_1) = 2 * (0 - 1) = -2
        expected = torch.tensor([[2.0], [-2.0]])
        assert torch.allclose(deriv.control_points, expected)

    def test_cubic_derivative_is_quadratic(self):
        """Derivative of cubic Bezier is quadratic."""
        control_points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        deriv = bezier_derivative(curve)

        assert deriv.degree == 2
        assert deriv.control_points.shape == (3, 1)

        # Q_i = 3 * (P_{i+1} - P_i) = 3 * 1 = 3 for all i
        expected = torch.tensor([[3.0], [3.0], [3.0]])
        assert torch.allclose(deriv.control_points, expected)

    def test_derivative_of_constant_raises_error(self):
        """Cannot compute derivative of degree-0 curve."""
        control_points = torch.tensor([[1.0, 2.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        with pytest.raises(ValueError):
            bezier_derivative(curve)


class TestBezierDerivativeEvaluate:
    """Tests for bezier_derivative_evaluate function."""

    def test_evaluate_first_derivative(self):
        """Should evaluate first derivative correctly."""
        # Linear: P(t) = (1-t)*P0 + t*P1, P'(t) = P1 - P0 = [1, 2]
        control_points = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        t = torch.tensor([0.0, 0.5, 1.0])
        deriv_vals = bezier_derivative_evaluate(curve, t, order=1)

        # Constant derivative
        expected = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        assert torch.allclose(deriv_vals, expected)

    def test_evaluate_second_derivative(self):
        """Should evaluate second derivative correctly."""
        # Quadratic: has constant second derivative
        control_points = torch.tensor([[0.0], [1.0], [0.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        t = torch.tensor([0.0, 0.5, 1.0])
        deriv_vals = bezier_derivative_evaluate(curve, t, order=2)

        # First derivative: Q_i = 2 * (P_{i+1} - P_i)
        # Q_0 = 2 * (1 - 0) = 2
        # Q_1 = 2 * (0 - 1) = -2
        # Second derivative: R_0 = 1 * (Q_1 - Q_0) = 1 * (-2 - 2) = -4
        expected = torch.tensor([[-4.0], [-4.0], [-4.0]])
        assert torch.allclose(deriv_vals, expected)

    def test_invalid_order_raises_error(self):
        """Should raise error for invalid derivative order."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        with pytest.raises(ValueError):
            bezier_derivative_evaluate(curve, torch.tensor(0.5), order=0)

        with pytest.raises(ValueError):
            bezier_derivative_evaluate(curve, torch.tensor(0.5), order=2)


class TestBezierSplit:
    """Tests for bezier_split function."""

    def test_split_at_midpoint(self):
        """Split at t=0.5 should produce two curves."""
        control_points = torch.tensor([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        left, right = bezier_split(curve, 0.5)

        # Both should have same degree
        assert left.degree == curve.degree
        assert right.degree == curve.degree

        # Left curve at t=1 should equal original at t=0.5
        left_end = bezier_evaluate(left, torch.tensor(1.0))
        original_mid = bezier_evaluate(curve, torch.tensor(0.5))
        assert torch.allclose(left_end, original_mid)

        # Right curve at t=0 should equal original at t=0.5
        right_start = bezier_evaluate(right, torch.tensor(0.0))
        assert torch.allclose(right_start, original_mid)

    def test_split_linear(self):
        """Split linear Bezier."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        left, right = bezier_split(curve, 0.5)

        assert left.degree == 1
        assert right.degree == 1

        # Left: [0, 0.5], Right: [0.5, 1]
        assert torch.allclose(
            left.control_points, torch.tensor([[0.0], [0.5]])
        )
        assert torch.allclose(
            right.control_points, torch.tensor([[0.5], [1.0]])
        )

    def test_split_preserves_continuity(self):
        """Split curves should be C0 continuous at split point."""
        control_points = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
        )
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        for t_split in [0.25, 0.5, 0.75]:
            left, right = bezier_split(curve, t_split)

            left_end = bezier_evaluate(left, torch.tensor(1.0))
            right_start = bezier_evaluate(right, torch.tensor(0.0))
            original_at_split = bezier_evaluate(curve, torch.tensor(t_split))

            assert torch.allclose(left_end, original_at_split, atol=1e-6)
            assert torch.allclose(right_start, original_at_split, atol=1e-6)
            assert torch.allclose(left_end, right_start, atol=1e-6)

    def test_split_tensor_parameter(self):
        """Should accept tensor parameter."""
        control_points = torch.tensor([[0.0], [1.0]])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        t = torch.tensor(0.3)
        left, right = bezier_split(curve, t)

        assert torch.allclose(
            left.control_points, torch.tensor([[0.0], [0.3]])
        )
        assert torch.allclose(
            right.control_points, torch.tensor([[0.3], [1.0]])
        )


class TestBezierConvenience:
    """Tests for bezier convenience function."""

    def test_bezier_returns_callable(self):
        """bezier() should return a callable."""
        control_points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        f = bezier(control_points)

        assert callable(f)

        result = f(torch.tensor([0.0, 0.5, 1.0]))
        assert result.shape == (3, 2)

    def test_bezier_extrapolate_option(self):
        """Should pass extrapolate option to curve."""
        control_points = torch.tensor([[0.0], [1.0]])

        f_error = bezier(control_points, extrapolate="error")
        f_clamp = bezier(control_points, extrapolate="clamp")

        with pytest.raises(ExtrapolationError):
            f_error(torch.tensor(-0.1))

        # Should not raise
        f_clamp(torch.tensor(-0.1))


class TestBezierHighDegree:
    """Tests for high-degree Bezier curves."""

    def test_degree_5_curve(self):
        """Should handle degree 5 curve."""
        control_points = torch.tensor([[float(i)] for i in range(6)])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        assert curve.degree == 5

        # Evaluate at endpoints
        assert torch.allclose(
            bezier_evaluate(curve, torch.tensor(0.0)),
            torch.tensor([0.0]),
        )
        assert torch.allclose(
            bezier_evaluate(curve, torch.tensor(1.0)),
            torch.tensor([5.0]),
        )

    def test_high_degree_derivative(self):
        """Should handle derivatives of high-degree curves."""
        control_points = torch.tensor([[float(i)] for i in range(6)])
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        for order in range(1, 6):
            deriv_vals = bezier_derivative_evaluate(
                curve, torch.tensor(0.5), order=order
            )
            assert deriv_vals.shape == (1,)


class TestBezier3D:
    """Tests for 3D Bezier curves."""

    def test_3d_curve_evaluation(self):
        """Should handle 3D control points."""
        control_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        t = torch.tensor([0.0, 0.5, 1.0])
        result = bezier_evaluate(curve, t)

        assert result.shape == (3, 3)

        # Endpoints
        assert torch.allclose(result[0], control_points[0])
        assert torch.allclose(result[2], control_points[2])

    def test_3d_derivative(self):
        """Should compute derivatives for 3D curves."""
        control_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        curve = BezierCurve(
            control_points=control_points, extrapolate="error", batch_size=[]
        )

        deriv = bezier_derivative(curve)

        assert deriv.control_points.shape == (1, 3)
        assert torch.allclose(
            deriv.control_points, torch.tensor([[1.0, 1.0, 1.0]])
        )


class TestBezierGradients:
    """Tests for gradient computation through Bezier operations."""

    def test_gradcheck_control_points(self):
        """Gradient check with respect to control points."""
        control_points = torch.tensor(
            [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        curve = BezierCurve(
            control_points=control_points, extrapolate="clamp", batch_size=[]
        )

        t = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
        result = bezier_evaluate(curve, t)
        loss = result.sum()
        loss.backward()

        assert control_points.grad is not None
        assert control_points.grad.shape == control_points.shape
