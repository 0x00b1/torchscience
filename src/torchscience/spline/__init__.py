"""Differentiable spline interpolation for PyTorch tensors.

This module provides cubic splines, B-splines, PCHIP, Hermite splines,
Bezier curves, Catmull-Rom splines, and smoothing splines with full autograd support.

Convenience Functions
---------------------
cubic_spline
    Create a cubic spline interpolator from data (fit + callable).
b_spline
    Create a B-spline approximation from data (fit + callable).
pchip
    Create a PCHIP interpolator from data (fit + callable).
hermite_spline
    Create a Hermite spline with user-specified derivatives.
bezier
    Create a Bezier curve from control points.
catmull_rom
    Create a Catmull-Rom spline from control points.
smoothing_spline
    Create a smoothing spline for noisy data.

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
b_spline_integral
    Compute definite integral of a B-spline.
b_spline_antiderivative
    Compute antiderivative B-spline (degree + 1).
b_spline_basis
    Evaluate B-spline basis functions.

PCHIP
-----
pchip_fit
    Fit a PCHIP spline to data points (monotonicity preserving).
pchip_evaluate
    Evaluate a PCHIP spline at query points.
pchip_derivative
    Compute derivatives of a PCHIP spline.
pchip_integral
    Compute definite integral of a PCHIP spline.

Hermite Splines
---------------
hermite_spline_fit
    Fit a Hermite spline with user-specified derivatives.
hermite_spline_evaluate
    Evaluate a Hermite spline at query points.
hermite_spline_derivative
    Compute derivatives of a Hermite spline.
hermite_spline_integral
    Compute definite integral of a Hermite spline.

Bezier Curves
-------------
bezier
    Create a Bezier curve from control points.
bezier_evaluate
    Evaluate a Bezier curve at parameter values.
bezier_derivative
    Compute the derivative Bezier curve.
bezier_split
    Split a Bezier curve at a parameter value.

Catmull-Rom Splines
-------------------
catmull_rom
    Create a Catmull-Rom spline from control points.
catmull_rom_evaluate
    Evaluate a Catmull-Rom spline at parameter values.
catmull_rom_derivative
    Compute derivatives of a Catmull-Rom spline.

Smoothing Splines
-----------------
smoothing_spline
    Create a smoothing spline for noisy data.
smoothing_spline_fit
    Fit a smoothing spline with automatic or manual smoothing parameter.
smoothing_spline_evaluate
    Evaluate a smoothing spline at query points.
smoothing_spline_derivative
    Compute derivatives of a smoothing spline.

Tensor Product Splines
----------------------
tensor_product_spline
    Create a 2D tensor product spline for gridded data.
tensor_product_fit
    Fit a tensor product spline to gridded data.
tensor_product_evaluate
    Evaluate a tensor product spline at query points.
tensor_product_derivative
    Compute partial derivatives of a tensor product spline.

RBF Interpolation
-----------------
rbf_interpolate
    Create an RBF interpolator for scattered data.
rbf_fit
    Fit an RBF interpolator to scattered data.
rbf_evaluate
    Evaluate an RBF interpolator at query points.

Analysis Operations
-------------------
spline_roots
    Find roots of a cubic spline.
spline_extrema
    Find local extrema of a cubic spline.
spline_minima
    Find local minima of a cubic spline.
spline_maxima
    Find local maxima of a cubic spline.
spline_arc_length
    Compute arc length of a spline curve.
spline_curvature
    Compute curvature at points on a spline.

Spline Arithmetic
-----------------
spline_add
    Add two B-splines.
spline_subtract
    Subtract two B-splines.
spline_scale
    Scale a B-spline by a constant.
spline_multiply
    Multiply two B-splines.
spline_negate
    Negate a B-spline.
spline_compose
    Compose two B-splines.

Data Types
----------
CubicSpline
    Piecewise cubic polynomial interpolant.
BSpline
    B-spline curve.
PCHIPSpline
    Piecewise Cubic Hermite Interpolating Polynomial.
HermiteSpline
    Cubic Hermite spline with user-specified derivatives.
BezierCurve
    Bezier curve defined by control points.
CatmullRomSpline
    Catmull-Rom spline that passes through control points.
SmoothingSpline
    Cubic smoothing spline for noisy data.
TensorProductSpline
    Bicubic tensor product spline for 2D interpolation.
RBFInterpolator
    Radial basis function interpolator for scattered data.

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

# Import analysis operations
from ._analysis import (
    spline_arc_length,
    spline_arc_length_parametric,
    spline_curvature,
    spline_curvature_parametric,
    spline_extrema,
    spline_maxima,
    spline_minima,
    spline_radius_of_curvature,
    spline_roots,
)

# Import arithmetic operations
from ._arithmetic import (
    spline_add,
    spline_compose,
    spline_multiply,
    spline_negate,
    spline_scale,
    spline_subtract,
)

# Import spline implementations
from ._b_spline import (
    BSpline,
    b_spline,
    b_spline_antiderivative,
    b_spline_basis,
    b_spline_derivative,
    b_spline_evaluate,
    b_spline_fit,
    b_spline_integral,
)
from ._bezier import (
    BezierCurve,
    bezier,
    bezier_derivative,
    bezier_derivative_evaluate,
    bezier_evaluate,
    bezier_split,
)
from ._catmull_rom import (
    CatmullRomSpline,
    catmull_rom,
    catmull_rom_derivative,
    catmull_rom_evaluate,
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
from ._hermite_spline import (
    HermiteSpline,
    hermite_spline,
    hermite_spline_derivative,
    hermite_spline_derivative_evaluate,
    hermite_spline_evaluate,
    hermite_spline_fit,
    hermite_spline_integral,
)
from ._knot_error import KnotError
from ._pchip import (
    PCHIPSpline,
    pchip,
    pchip_derivative,
    pchip_evaluate,
    pchip_fit,
    pchip_integral,
)
from ._rbf import (
    RBFInterpolator,
    rbf_evaluate,
    rbf_fit,
    rbf_interpolate,
)
from ._smoothing_spline import (
    SmoothingSpline,
    smoothing_spline,
    smoothing_spline_derivative,
    smoothing_spline_evaluate,
    smoothing_spline_fit,
)
from ._spline_error import SplineError
from ._tensor_product import (
    TensorProductSpline,
    tensor_product_derivative,
    tensor_product_evaluate,
    tensor_product_fit,
    tensor_product_spline,
)

__all__ = [
    "BSpline",
    "BezierCurve",
    "CatmullRomSpline",
    "CubicSpline",
    "DegreeError",
    "ExtrapolationError",
    "HermiteSpline",
    "KnotError",
    "PCHIPSpline",
    "RBFInterpolator",
    "SmoothingSpline",
    "SplineError",
    "TensorProductSpline",
    "b_spline",
    "b_spline_antiderivative",
    "b_spline_basis",
    "b_spline_derivative",
    "b_spline_evaluate",
    "b_spline_fit",
    "b_spline_integral",
    "bezier",
    "bezier_derivative",
    "bezier_derivative_evaluate",
    "bezier_evaluate",
    "bezier_split",
    "catmull_rom",
    "catmull_rom_derivative",
    "catmull_rom_evaluate",
    "cubic_spline",
    "cubic_spline_derivative",
    "cubic_spline_evaluate",
    "cubic_spline_fit",
    "cubic_spline_integral",
    "hermite_spline",
    "hermite_spline_derivative",
    "hermite_spline_derivative_evaluate",
    "hermite_spline_evaluate",
    "hermite_spline_fit",
    "hermite_spline_integral",
    "pchip",
    "pchip_derivative",
    "pchip_evaluate",
    "pchip_fit",
    "pchip_integral",
    "rbf_evaluate",
    "rbf_fit",
    "rbf_interpolate",
    "smoothing_spline",
    "smoothing_spline_derivative",
    "smoothing_spline_evaluate",
    "smoothing_spline_fit",
    "spline_add",
    "spline_arc_length",
    "spline_arc_length_parametric",
    "spline_compose",
    "spline_curvature",
    "spline_curvature_parametric",
    "spline_extrema",
    "spline_maxima",
    "spline_minima",
    "spline_multiply",
    "spline_negate",
    "spline_radius_of_curvature",
    "spline_roots",
    "spline_scale",
    "spline_subtract",
    "tensor_product_derivative",
    "tensor_product_evaluate",
    "tensor_product_fit",
    "tensor_product_spline",
]
