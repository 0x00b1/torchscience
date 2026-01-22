"""Spline analysis operations."""

from ._spline_arc_length import spline_arc_length, spline_arc_length_parametric
from ._spline_curvature import (
    spline_curvature,
    spline_curvature_parametric,
    spline_radius_of_curvature,
)
from ._spline_extrema import spline_extrema, spline_maxima, spline_minima
from ._spline_roots import spline_roots

__all__ = [
    "spline_arc_length",
    "spline_arc_length_parametric",
    "spline_curvature",
    "spline_curvature_parametric",
    "spline_extrema",
    "spline_maxima",
    "spline_minima",
    "spline_radius_of_curvature",
    "spline_roots",
]
