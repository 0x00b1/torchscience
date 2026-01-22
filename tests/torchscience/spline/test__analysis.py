"""Tests for spline analysis operations."""

import pytest
import torch

from torchscience.spline import (
    cubic_spline_fit,
    spline_arc_length,
    spline_curvature,
    spline_extrema,
    spline_maxima,
    spline_minima,
    spline_roots,
)


class TestSplineRoots:
    """Tests for spline_roots function."""

    def test_find_single_root(self):
        """Should find a single root."""
        # Linear function crossing zero
        x = torch.linspace(-1, 1, 5)
        y = x.clone()

        spline = cubic_spline_fit(x, y)
        roots = spline_roots(spline, value=0.0)

        assert roots.numel() == 1
        assert torch.allclose(roots, torch.tensor([0.0]), atol=1e-5)

    def test_find_multiple_roots(self):
        """Should find multiple roots."""
        # Sine-like function with multiple zeros
        x = torch.linspace(0, 2 * torch.pi, 20)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)
        roots = spline_roots(spline, value=0.0)

        # Should find roots near 0, pi, 2*pi
        assert roots.numel() >= 2

        # Check that roots are approximately correct
        expected = torch.tensor([0.0, torch.pi, 2 * torch.pi])
        for exp_root in expected:
            diffs = (roots - exp_root).abs()
            assert diffs.min() < 0.2  # Allow some tolerance

    def test_find_roots_at_value(self):
        """Should find roots at non-zero value."""
        x = torch.linspace(0, 1, 10)
        y = x**2

        spline = cubic_spline_fit(x, y)
        roots = spline_roots(spline, value=0.25)

        assert roots.numel() == 1
        assert torch.allclose(roots, torch.tensor([0.5]), atol=0.1)

    def test_no_roots(self):
        """Should return empty tensor when no roots."""
        x = torch.linspace(0, 1, 10)
        y = x + 1  # Always positive

        spline = cubic_spline_fit(x, y)
        roots = spline_roots(spline, value=0.0)

        assert roots.numel() == 0


class TestSplineExtrema:
    """Tests for spline_extrema function."""

    def test_find_single_extremum(self):
        """Should find a single extremum."""
        # Parabola with minimum at x=0.5
        x = torch.linspace(0, 1, 10)
        y = (x - 0.5) ** 2

        spline = cubic_spline_fit(x, y)
        x_ext, y_ext = spline_extrema(spline)

        assert x_ext.numel() == 1
        assert torch.allclose(x_ext, torch.tensor([0.5]), atol=0.1)
        assert torch.allclose(y_ext, torch.tensor([0.0]), atol=0.05)

    def test_find_multiple_extrema(self):
        """Should find multiple extrema."""
        # Sine with multiple extrema
        x = torch.linspace(0, 2 * torch.pi, 30)
        y = torch.sin(x)

        spline = cubic_spline_fit(x, y)
        x_ext, y_ext = spline_extrema(spline)

        # Should find extrema near pi/2 and 3*pi/2
        assert x_ext.numel() >= 2

    def test_no_extrema(self):
        """Should return empty for monotonic function."""
        x = torch.linspace(0, 1, 10)
        y = x.clone()  # Strictly increasing

        spline = cubic_spline_fit(x, y)
        x_ext, y_ext = spline_extrema(spline)

        assert x_ext.numel() == 0
        assert y_ext.numel() == 0


class TestSplineMinMax:
    """Tests for spline_minima and spline_maxima functions."""

    def test_find_minima(self):
        """Should find local minima."""
        # Function with a minimum
        x = torch.linspace(0, 1, 10)
        y = (x - 0.5) ** 2

        spline = cubic_spline_fit(x, y)
        x_min, y_min = spline_minima(spline)

        assert x_min.numel() == 1
        assert torch.allclose(x_min, torch.tensor([0.5]), atol=0.1)

    def test_find_maxima(self):
        """Should find local maxima."""
        # Function with a maximum
        x = torch.linspace(0, 1, 10)
        y = -((x - 0.5) ** 2)

        spline = cubic_spline_fit(x, y)
        x_max, y_max = spline_maxima(spline)

        assert x_max.numel() == 1
        assert torch.allclose(x_max, torch.tensor([0.5]), atol=0.1)


class TestSplineArcLength:
    """Tests for spline_arc_length function."""

    def test_arc_length_straight_line(self):
        """Arc length of straight line should be line length."""
        x = torch.linspace(0, 1, 10)
        y = x.clone()  # y = x, length = sqrt(2)

        spline = cubic_spline_fit(x, y)
        length = spline_arc_length(spline)

        expected = torch.sqrt(torch.tensor(2.0))
        assert torch.allclose(length, expected, rtol=0.05)

    def test_arc_length_partial(self):
        """Should compute arc length over partial interval."""
        x = torch.linspace(0, 1, 10)
        y = x.clone()

        spline = cubic_spline_fit(x, y)

        # Half the interval should give half the length
        full_length = spline_arc_length(spline)
        half_length = spline_arc_length(
            spline, a=torch.tensor(0.0), b=torch.tensor(0.5)
        )

        assert torch.allclose(half_length, full_length / 2, rtol=0.1)

    def test_arc_length_horizontal(self):
        """Arc length of horizontal line should be interval length."""
        x = torch.linspace(0, 2, 10)
        y = torch.ones_like(x)  # Horizontal line

        spline = cubic_spline_fit(x, y)
        length = spline_arc_length(spline)

        # Length should be close to 2
        assert torch.allclose(length, torch.tensor(2.0), rtol=0.05)


class TestSplineCurvature:
    """Tests for spline_curvature function."""

    def test_curvature_straight_line(self):
        """Curvature of straight line should be zero."""
        x = torch.linspace(0, 1, 10)
        y = x.clone()

        spline = cubic_spline_fit(x, y)

        t = torch.tensor([0.25, 0.5, 0.75])
        kappa = spline_curvature(spline, t)

        assert torch.allclose(kappa, torch.zeros_like(kappa), atol=1e-3)

    def test_curvature_parabola(self):
        """Curvature of parabola should be highest at vertex."""
        x = torch.linspace(-1, 1, 20)
        y = x**2

        spline = cubic_spline_fit(x, y)

        # At vertex (x=0), curvature is 2 / (1 + 0)^1.5 = 2
        t = torch.tensor([0.0])
        kappa = spline_curvature(spline, t)

        assert torch.allclose(kappa, torch.tensor([2.0]), rtol=0.3)

    def test_curvature_batch(self):
        """Should handle batch queries."""
        x = torch.linspace(0, 1, 10)
        y = x**2

        spline = cubic_spline_fit(x, y)

        t = torch.linspace(0.1, 0.9, 5)
        kappa = spline_curvature(spline, t)

        assert kappa.shape == (5,)


class TestAnalysisEdgeCases:
    """Tests for edge cases in analysis functions."""

    def test_roots_multivalued_error(self):
        """Should raise error for multi-valued splines."""
        x = torch.linspace(0, 1, 10)
        y = torch.stack([x, x**2], dim=-1)  # 2D values

        spline = cubic_spline_fit(x, y)

        with pytest.raises(ValueError):
            spline_roots(spline)

    def test_extrema_multivalued_error(self):
        """Should raise error for multi-valued splines."""
        x = torch.linspace(0, 1, 10)
        y = torch.stack([x, x**2], dim=-1)

        spline = cubic_spline_fit(x, y)

        with pytest.raises(ValueError):
            spline_extrema(spline)
