from .._degree_error import DegreeError
from .__single_derivative import _single_derivative
from ._b_spline import BSpline


def b_spline_derivative(
    spline: BSpline,
    order: int = 1,
) -> BSpline:
    """
    Compute the derivative of a B-spline.

    Parameters
    ----------
    spline : BSpline
        Input B-spline
    order : int
        Order of derivative (default 1)

    Returns
    -------
    derivative : BSpline
        Derivative B-spline with degree = original_degree - order

    Raises
    ------
    DegreeError
        If derivative order is greater than the spline degree.

    Notes
    -----
    The derivative of a degree-k B-spline is a degree-(k-1) B-spline.
    The new control points are computed as:
        d_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
    where c_i are original control points and t_i are knots.

    The new knot vector has the first and last knots removed.

    For higher order derivatives (order > 1), the formula is applied recursively.
    """
    if order < 1:
        raise ValueError(f"Derivative order must be at least 1, got {order}")

    if order > spline.degree:
        raise DegreeError(
            f"Cannot compute order-{order} derivative of degree-{spline.degree} spline"
        )

    # Apply derivative formula recursively for higher orders
    current_spline = spline
    for _ in range(order):
        current_spline = _single_derivative(current_spline)

    return current_spline
