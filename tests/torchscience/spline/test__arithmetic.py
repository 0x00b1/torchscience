"""Tests for spline arithmetic operations."""

import pytest
import torch

from torchscience.spline import (
    b_spline_evaluate,
    b_spline_fit,
    spline_add,
    spline_compose,
    spline_multiply,
    spline_negate,
    spline_scale,
    spline_subtract,
)


class TestSplineAdd:
    """Tests for spline_add function."""

    def test_add_basic(self):
        """Should add two B-splines."""
        x = torch.linspace(0, 1, 10)
        y1 = x.clone()
        y2 = x**2

        s1 = b_spline_fit(x, y1, degree=3)
        s2 = b_spline_fit(x, y2, degree=3)

        result = spline_add(s1, s2)

        # Check at some points
        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = t + t**2

        assert torch.allclose(val, expected, rtol=0.1, atol=0.05)

    def test_add_commutative(self):
        """Addition should be commutative."""
        x = torch.linspace(0, 1, 10)
        s1 = b_spline_fit(x, x, degree=3)
        s2 = b_spline_fit(x, x**2, degree=3)

        result1 = spline_add(s1, s2)
        result2 = spline_add(s2, s1)

        t = torch.linspace(0.1, 0.9, 5)
        v1 = b_spline_evaluate(result1, t)
        v2 = b_spline_evaluate(result2, t)

        assert torch.allclose(v1, v2, atol=0.05)


class TestSplineSubtract:
    """Tests for spline_subtract function."""

    def test_subtract_basic(self):
        """Should subtract two B-splines."""
        x = torch.linspace(0, 1, 10)
        y1 = x**2
        y2 = x.clone()

        s1 = b_spline_fit(x, y1, degree=3)
        s2 = b_spline_fit(x, y2, degree=3)

        result = spline_subtract(s1, s2)

        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = t**2 - t

        assert torch.allclose(val, expected, rtol=0.1, atol=0.05)

    def test_subtract_self_is_zero(self):
        """Subtracting a spline from itself should give zero."""
        x = torch.linspace(0, 1, 10)
        s = b_spline_fit(x, x**2, degree=3)

        result = spline_subtract(s, s)

        t = torch.linspace(0.1, 0.9, 5)
        val = b_spline_evaluate(result, t)

        assert torch.allclose(val, torch.zeros_like(val), atol=0.05)


class TestSplineScale:
    """Tests for spline_scale function."""

    def test_scale_basic(self):
        """Should scale a B-spline by constant."""
        x = torch.linspace(0, 1, 10)
        s = b_spline_fit(x, x**2, degree=3)

        result = spline_scale(s, 3.0)

        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = 3.0 * t**2

        assert torch.allclose(val, expected, rtol=0.1, atol=0.05)

    def test_scale_by_zero(self):
        """Scaling by zero should give zero spline."""
        x = torch.linspace(0, 1, 10)
        s = b_spline_fit(x, x**2, degree=3)

        result = spline_scale(s, 0.0)

        t = torch.linspace(0.1, 0.9, 5)
        val = b_spline_evaluate(result, t)

        assert torch.allclose(val, torch.zeros_like(val), atol=1e-5)

    def test_scale_by_one(self):
        """Scaling by one should preserve spline."""
        x = torch.linspace(0, 1, 10)
        s = b_spline_fit(x, x**2, degree=3)

        result = spline_scale(s, 1.0)

        t = torch.linspace(0.1, 0.9, 5)
        v_orig = b_spline_evaluate(s, t)
        v_scaled = b_spline_evaluate(result, t)

        assert torch.allclose(v_orig, v_scaled, atol=1e-5)


class TestSplineNegate:
    """Tests for spline_negate function."""

    def test_negate_basic(self):
        """Should negate a B-spline."""
        x = torch.linspace(0, 1, 10)
        s = b_spline_fit(x, x, degree=3)

        result = spline_negate(s)

        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = -t

        assert torch.allclose(val, expected, rtol=0.1, atol=0.05)

    def test_double_negate(self):
        """Double negation should restore original."""
        x = torch.linspace(0, 1, 10)
        s = b_spline_fit(x, x**2, degree=3)

        result = spline_negate(spline_negate(s))

        t = torch.linspace(0.1, 0.9, 5)
        v_orig = b_spline_evaluate(s, t)
        v_double = b_spline_evaluate(result, t)

        assert torch.allclose(v_orig, v_double, atol=1e-5)


class TestSplineMultiply:
    """Tests for spline_multiply function."""

    def test_multiply_basic(self):
        """Should multiply two B-splines."""
        x = torch.linspace(0, 1, 15)
        s1 = b_spline_fit(x, x, degree=3)
        s2 = b_spline_fit(x, x, degree=3)

        result = spline_multiply(s1, s2)

        # x * x = x^2
        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = t**2

        assert torch.allclose(val, expected, rtol=0.2, atol=0.1)

    def test_multiply_by_constant(self):
        """Multiplying by constant spline should scale."""
        x = torch.linspace(0, 1, 15)
        s = b_spline_fit(x, x**2, degree=3)
        s_const = b_spline_fit(x, torch.full_like(x, 2.0), degree=3)

        result = spline_multiply(s, s_const)

        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = 2.0 * t**2

        assert torch.allclose(val, expected, rtol=0.2, atol=0.1)


class TestSplineCompose:
    """Tests for spline_compose function."""

    def test_compose_identity(self):
        """Composing with identity should preserve function."""
        x = torch.linspace(0, 1, 15)
        s_outer = b_spline_fit(x, x**2, degree=3)
        s_identity = b_spline_fit(x, x, degree=3)

        result = spline_compose(s_outer, s_identity)

        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = t**2

        assert torch.allclose(val, expected, rtol=0.2, atol=0.1)

    def test_compose_basic(self):
        """Should compose two B-splines."""
        x = torch.linspace(0, 1, 15)

        # outer(t) = t^2
        # inner(t) = 0.5 * t
        # result(t) = outer(inner(t)) = (0.5*t)^2 = 0.25*t^2
        s_outer = b_spline_fit(x, x**2, degree=3)
        s_inner = b_spline_fit(x, 0.5 * x, degree=3)

        result = spline_compose(s_outer, s_inner)

        t = torch.tensor([0.25, 0.5, 0.75])
        val = b_spline_evaluate(result, t)
        expected = 0.25 * t**2

        assert torch.allclose(val, expected, rtol=0.3, atol=0.1)


class TestSplineArithmeticEdgeCases:
    """Tests for edge cases in spline arithmetic."""

    def test_non_overlapping_domains(self):
        """Should raise error for non-overlapping domains."""
        x1 = torch.linspace(0, 1, 10)
        x2 = torch.linspace(2, 3, 10)

        s1 = b_spline_fit(x1, x1, degree=3)
        s2 = b_spline_fit(x2, x2, degree=3)

        with pytest.raises(ValueError, match="non-overlapping"):
            spline_add(s1, s2)

    def test_partial_overlap(self):
        """Should work with partially overlapping domains."""
        x1 = torch.linspace(0, 2, 15)
        x2 = torch.linspace(1, 3, 15)

        s1 = b_spline_fit(x1, x1, degree=3)
        s2 = b_spline_fit(x2, x2, degree=3)

        result = spline_add(s1, s2)

        # Should be defined on [1, 2]
        t = torch.tensor([1.25, 1.5, 1.75])
        val = b_spline_evaluate(result, t)
        expected = 2 * t  # x + x = 2x

        assert torch.allclose(val, expected, rtol=0.2, atol=0.1)
