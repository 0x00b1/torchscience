"""Tests for Catmull-Rom spline functions."""

import pytest
import torch

from torchscience.spline import (
    CatmullRomSpline,
    ExtrapolationError,
    catmull_rom,
    catmull_rom_derivative,
    catmull_rom_evaluate,
)


class TestCatmullRomSpline:
    """Tests for CatmullRomSpline tensorclass."""

    def test_creation(self):
        """Should create spline with correct attributes."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        assert spline.control_points.shape == (4, 1)
        assert spline.alpha == 0.5
        assert spline.extrapolate == "error"


class TestCatmullRomEvaluate:
    """Tests for catmull_rom_evaluate function."""

    def test_passes_through_interior_points(self):
        """Spline should pass through interior control points."""
        # 4 control points means the curve passes through points[1] and points[2]
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        # At t=0, curve passes through points[1]
        result_0 = catmull_rom_evaluate(spline, torch.tensor(0.0))
        assert torch.allclose(result_0, points[1], atol=1e-5)

        # At t=1, curve passes through points[2]
        result_1 = catmull_rom_evaluate(spline, torch.tensor(1.0))
        assert torch.allclose(result_1, points[2], atol=1e-5)

    def test_uniform_parameterization(self):
        """Uniform (alpha=0) should work."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.0,
            extrapolate="error",
            batch_size=[],
        )

        # Should still pass through interior points
        result_0 = catmull_rom_evaluate(spline, torch.tensor(0.0))
        result_1 = catmull_rom_evaluate(spline, torch.tensor(1.0))

        assert torch.allclose(result_0, points[1], atol=1e-5)
        assert torch.allclose(result_1, points[2], atol=1e-5)

    def test_centripetal_parameterization(self):
        """Centripetal (alpha=0.5) should work."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        t = torch.linspace(0, 1, 10)
        result = catmull_rom_evaluate(spline, t)

        assert result.shape == (10, 2)

    def test_chordal_parameterization(self):
        """Chordal (alpha=1.0) should work."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=1.0,
            extrapolate="error",
            batch_size=[],
        )

        t = torch.linspace(0, 1, 10)
        result = catmull_rom_evaluate(spline, t)

        assert result.shape == (10, 2)

    def test_multiple_segments(self):
        """Should handle curves with multiple segments."""
        # 6 control points -> 3 segments (parameter range [0, 3])
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        # Check that we can evaluate across all segments
        t = torch.linspace(0, 3, 20)
        result = catmull_rom_evaluate(spline, t)

        assert result.shape == (20, 1)

        # Should pass through interior points at integer parameters
        for i in range(4):  # points[1], points[2], points[3], points[4]
            result_i = catmull_rom_evaluate(spline, torch.tensor(float(i)))
            assert torch.allclose(result_i, points[i + 1], atol=1e-5)

    def test_scalar_query(self):
        """Should handle scalar parameter."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        result = catmull_rom_evaluate(spline, torch.tensor(0.5))
        assert result.shape == (1,)

    def test_batch_query(self):
        """Should handle batched parameters."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        t = torch.tensor([[0.0, 0.5], [0.75, 1.0]])  # Shape (2, 2)
        result = catmull_rom_evaluate(spline, t)

        assert result.shape == (2, 2, 2)  # (2, 2, value_dim)

    def test_extrapolate_error(self):
        """Should raise error for out-of-range parameters."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        with pytest.raises(ExtrapolationError):
            catmull_rom_evaluate(spline, torch.tensor(-0.1))

        with pytest.raises(ExtrapolationError):
            catmull_rom_evaluate(spline, torch.tensor(1.1))

    def test_extrapolate_clamp(self):
        """Should clamp parameters to valid range."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="clamp",
            batch_size=[],
        )

        result_neg = catmull_rom_evaluate(spline, torch.tensor(-0.5))
        result_pos = catmull_rom_evaluate(spline, torch.tensor(1.5))

        # Should be clamped to endpoints
        assert torch.allclose(result_neg, points[1], atol=1e-5)
        assert torch.allclose(result_pos, points[2], atol=1e-5)

    def test_extrapolate_extend(self):
        """Should allow extrapolation."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="extrapolate",
            batch_size=[],
        )

        # Should not raise
        result_neg = catmull_rom_evaluate(spline, torch.tensor(-0.1))
        result_pos = catmull_rom_evaluate(spline, torch.tensor(1.1))

        assert result_neg.shape == (1,)
        assert result_pos.shape == (1,)

    def test_1d_values(self):
        """Should handle 1D value tensors."""
        points = torch.tensor([0.0, 1.0, 2.0, 3.0])  # Shape (4,)
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        result = catmull_rom_evaluate(spline, torch.tensor(0.5))
        assert result.dim() == 0 or result.shape == ()

    def test_3d_values(self):
        """Should handle 3D control points."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        t = torch.linspace(0, 1, 5)
        result = catmull_rom_evaluate(spline, t)

        assert result.shape == (5, 3)

    def test_gradcheck(self):
        """Gradient check for Catmull-Rom evaluation."""
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]],
            dtype=torch.float64,
        )
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="clamp",
            batch_size=[],
        )

        t = torch.tensor(
            [0.25, 0.5, 0.75], dtype=torch.float64, requires_grad=True
        )

        result = catmull_rom_evaluate(spline, t)
        loss = result.sum()
        loss.backward()

        assert t.grad is not None


class TestCatmullRomDerivative:
    """Tests for catmull_rom_derivative function."""

    def test_first_derivative(self):
        """Should compute first derivative."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        t = torch.tensor([0.0, 0.5, 1.0])
        deriv = catmull_rom_derivative(spline, t, order=1)

        assert deriv.shape == (3, 2)

    def test_second_derivative(self):
        """Should compute second derivative."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        t = torch.tensor([0.0, 0.5, 1.0])
        deriv = catmull_rom_derivative(spline, t, order=2)

        assert deriv.shape == (3, 2)

    def test_invalid_order(self):
        """Should raise error for invalid order."""
        points = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        with pytest.raises(ValueError):
            catmull_rom_derivative(spline, torch.tensor(0.5), order=0)

        with pytest.raises(ValueError):
            catmull_rom_derivative(spline, torch.tensor(0.5), order=3)

    def test_derivative_consistency(self):
        """Numerical derivative should match derivative function."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="extrapolate",
            batch_size=[],
        )

        t = torch.tensor([0.25, 0.5, 0.75])
        h = 1e-4

        # Numerical derivative
        f_plus = catmull_rom_evaluate(spline, t + h)
        f_minus = catmull_rom_evaluate(spline, t - h)
        numerical_deriv = (f_plus - f_minus) / (2 * h)

        # Function derivative
        deriv = catmull_rom_derivative(spline, t, order=1)

        assert torch.allclose(numerical_deriv, deriv, rtol=1e-2, atol=1e-3)


class TestCatmullRomConvenience:
    """Tests for catmull_rom convenience function."""

    def test_returns_callable(self):
        """catmull_rom() should return a callable."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
        f = catmull_rom(points)

        assert callable(f)

        result = f(torch.tensor([0.0, 0.5, 1.0]))
        assert result.shape == (3, 2)

    def test_alpha_parameter(self):
        """Should pass alpha to spline."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])

        f_uniform = catmull_rom(points, alpha=0.0)
        f_centripetal = catmull_rom(points, alpha=0.5)
        f_chordal = catmull_rom(points, alpha=1.0)

        t = torch.tensor(0.5)

        # All should work and may produce different results
        r_uniform = f_uniform(t)
        r_centripetal = f_centripetal(t)
        r_chordal = f_chordal(t)

        assert r_uniform.shape == (2,)
        assert r_centripetal.shape == (2,)
        assert r_chordal.shape == (2,)

    def test_extrapolate_parameter(self):
        """Should pass extrapolate to spline."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])

        f_error = catmull_rom(points, extrapolate="error")
        f_clamp = catmull_rom(points, extrapolate="clamp")

        with pytest.raises(ExtrapolationError):
            f_error(torch.tensor(-0.1))

        # Should not raise
        f_clamp(torch.tensor(-0.1))

    def test_requires_4_points(self):
        """Should require at least 4 control points."""
        points = torch.tensor([[0.0], [1.0], [2.0]])  # Only 3 points

        with pytest.raises(ValueError):
            catmull_rom(points)


class TestCatmullRomContinuity:
    """Tests for curve continuity properties."""

    def test_c1_continuity_at_knots(self):
        """Curve should be C1 continuous at interior knots."""
        # With 5 control points, we have 2 segments meeting at t=1
        points = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]]
        )
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="extrapolate",
            batch_size=[],
        )

        # Check continuity at t=1 (junction between segments 0 and 1)
        eps = 1e-5

        # Value continuity
        left_val = catmull_rom_evaluate(spline, torch.tensor(1.0 - eps))
        right_val = catmull_rom_evaluate(spline, torch.tensor(1.0 + eps))
        assert torch.allclose(left_val, right_val, rtol=1e-3, atol=1e-4)

        # Derivative continuity
        left_deriv = catmull_rom_derivative(spline, torch.tensor(1.0 - eps))
        right_deriv = catmull_rom_derivative(spline, torch.tensor(1.0 + eps))
        assert torch.allclose(left_deriv, right_deriv, rtol=1e-2, atol=1e-3)


class TestCatmullRomSpecialCases:
    """Tests for special cases and edge conditions."""

    def test_collinear_points(self):
        """Should handle collinear points."""
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        # For collinear points, curve should be linear
        t = torch.linspace(0, 1, 5)
        result = catmull_rom_evaluate(spline, t)

        # All points should be on the line y = x
        assert torch.allclose(result[:, 0], result[:, 1], atol=1e-5)

    def test_coincident_points(self):
        """Should handle nearly coincident points."""
        points = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        spline = CatmullRomSpline(
            control_points=points,
            alpha=0.5,
            extrapolate="error",
            batch_size=[],
        )

        # Should not crash
        t = torch.linspace(0, 1, 5)
        result = catmull_rom_evaluate(spline, t)

        assert result.shape == (5, 2)
        assert not torch.isnan(result).any()
