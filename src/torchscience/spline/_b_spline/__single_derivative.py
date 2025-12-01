import torch

from ._b_spline import BSpline


def _single_derivative(spline: BSpline) -> BSpline:
    """
    Compute a single derivative of a B-spline.

    Parameters
    ----------
    spline : BSpline
        Input B-spline

    Returns
    -------
    derivative : BSpline
        Derivative B-spline with degree = original_degree - 1
    """
    knots = spline.knots
    control_points = spline.control_points
    degree = spline.degree

    # New degree is one less
    new_degree = degree - 1

    # New knot vector: remove first and last knots
    new_knots = knots[1:-1]

    # Number of new control points: n_control - 1
    n_control = control_points.shape[0]
    n_new_control = n_control - 1

    # Compute new control points using the formula:
    # d_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
    # where k is the degree, c_i are original control points, t_i are knots
    #
    # Index mapping:
    # - For new control point d_i (i = 0, ..., n_new_control - 1):
    #   - Use c_i and c_{i+1}
    #   - Use t_{i+1} and t_{i+k+1} (where k = degree)

    # Handle multi-dimensional control points
    if control_points.dim() == 1:
        new_control_points = torch.zeros(
            n_new_control,
            dtype=control_points.dtype,
            device=control_points.device,
        )
    else:
        value_shape = control_points.shape[1:]
        new_control_points = torch.zeros(
            n_new_control,
            *value_shape,
            dtype=control_points.dtype,
            device=control_points.device,
        )

    for i in range(n_new_control):
        # Denominator: t_{i+k+1} - t_{i+1}
        denom = knots[i + degree + 1] - knots[i + 1]

        # Numerator: k * (c_{i+1} - c_i)
        diff = control_points[i + 1] - control_points[i]
        numerator = degree * diff

        # Handle 0/0 case (zero-length knot interval)
        if denom.abs() > 0:
            new_control_points[i] = numerator / denom
        else:
            # 0/0 is treated as 0
            new_control_points[i] = torch.zeros_like(diff)

    return BSpline(
        knots=new_knots,
        control_points=new_control_points,
        degree=new_degree,
        extrapolate=spline.extrapolate,
        batch_size=[],
    )
