"""Cubic Hermite spline module with user-specified derivatives."""

from ._hermite_spline import HermiteSpline, hermite_spline
from ._hermite_spline_derivative import (
    hermite_spline_derivative,
    hermite_spline_derivative_evaluate,
)
from ._hermite_spline_evaluate import hermite_spline_evaluate
from ._hermite_spline_fit import hermite_spline_fit
from ._hermite_spline_integral import hermite_spline_integral

__all__ = [
    "HermiteSpline",
    "hermite_spline",
    "hermite_spline_derivative",
    "hermite_spline_derivative_evaluate",
    "hermite_spline_evaluate",
    "hermite_spline_fit",
    "hermite_spline_integral",
]
