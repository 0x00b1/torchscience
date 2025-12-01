"""Differentiable spline interpolation for PyTorch tensors.

This module provides cubic splines and B-splines with full autograd support.

Convenience Functions
---------------------
cubic_spline
    Create a cubic spline interpolator from data (fit + callable).
b_spline
    Create a B-spline approximation from data (fit + callable).

Cubic Splines
-------------
cubic_spline_fit
    Fit a cubic spline to data points.
cubic_spline_evaluate
    Evaluate a cubic spline at query points.
cubic_spline_derivative
    Compute derivatives of a cubic spline.
cubic_spline_integral
    Compute definite integral of a cubic spline.

B-Splines
---------
b_spline_fit
    Fit a B-spline to data points.
b_spline_evaluate
    Evaluate a B-spline at query points.
b_spline_derivative
    Compute derivatives of a B-spline.
b_spline_basis
    Evaluate B-spline basis functions.

Data Types
----------
CubicSpline
    Piecewise cubic polynomial interpolant.
BSpline
    B-spline curve.

Exceptions
----------
SplineError
    Base exception for spline operations.
ExtrapolationError
    Query point outside spline domain.
KnotError
    Invalid knot vector.
DegreeError
    Invalid degree for given knots.
"""

# Import base exception first
# Import spline implementations
from ._b_spline import (
    BSpline,
    b_spline,
    b_spline_basis,
    b_spline_derivative,
    b_spline_evaluate,
    b_spline_fit,
)
from ._cubic_spline import (
    CubicSpline,
    cubic_spline,
    cubic_spline_derivative,
    cubic_spline_evaluate,
    cubic_spline_fit,
    cubic_spline_integral,
)

# Import exception subclasses
from ._degree_error import DegreeError
from ._extrapolation_error import ExtrapolationError
from ._knot_error import KnotError
from ._spline_error import SplineError

__all__ = [
    "BSpline",
    "CubicSpline",
    "DegreeError",
    "ExtrapolationError",
    "KnotError",
    "SplineError",
    "b_spline",
    "b_spline_basis",
    "b_spline_derivative",
    "b_spline_evaluate",
    "b_spline_fit",
    "cubic_spline",
    "cubic_spline_derivative",
    "cubic_spline_evaluate",
    "cubic_spline_fit",
    "cubic_spline_integral",
]
