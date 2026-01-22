"""Spline arithmetic operations."""

from __future__ import annotations

import torch

from ._b_spline import BSpline, b_spline_evaluate


def spline_add(s1: BSpline, s2: BSpline) -> BSpline:
    """
    Add two B-splines.

    Parameters
    ----------
    s1 : BSpline
        First B-spline.
    s2 : BSpline
        Second B-spline.

    Returns
    -------
    result : BSpline
        Sum of the two splines.

    Notes
    -----
    The result is defined on the intersection of the two domains.
    For efficiency, this uses sampling and re-fitting rather than
    exact knot insertion algorithms.
    """
    # Determine common domain
    domain_start = max(s1.knots[s1.degree].item(), s2.knots[s2.degree].item())
    domain_end = min(
        s1.knots[-(s1.degree + 1)].item(), s2.knots[-(s2.degree + 1)].item()
    )

    if domain_end <= domain_start:
        raise ValueError("Splines have non-overlapping domains")

    # Sample and add
    result_degree = max(s1.degree, s2.degree)
    n_points = max(s1.control_points.shape[0], s2.control_points.shape[0]) * 2

    t = torch.linspace(
        domain_start,
        domain_end,
        n_points,
        dtype=s1.control_points.dtype,
        device=s1.control_points.device,
    )

    y1 = b_spline_evaluate(s1, t)
    y2 = b_spline_evaluate(s2, t)
    y_sum = y1 + y2

    # Fit new B-spline to the sum
    from ._b_spline import b_spline_fit

    return b_spline_fit(
        t, y_sum, degree=result_degree, extrapolate=s1.extrapolate
    )


def spline_subtract(s1: BSpline, s2: BSpline) -> BSpline:
    """
    Subtract two B-splines (s1 - s2).

    Parameters
    ----------
    s1 : BSpline
        First B-spline.
    s2 : BSpline
        Second B-spline (subtracted).

    Returns
    -------
    result : BSpline
        Difference of the two splines.
    """
    # Determine common domain
    domain_start = max(s1.knots[s1.degree].item(), s2.knots[s2.degree].item())
    domain_end = min(
        s1.knots[-(s1.degree + 1)].item(), s2.knots[-(s2.degree + 1)].item()
    )

    if domain_end <= domain_start:
        raise ValueError("Splines have non-overlapping domains")

    result_degree = max(s1.degree, s2.degree)
    n_points = max(s1.control_points.shape[0], s2.control_points.shape[0]) * 2

    t = torch.linspace(
        domain_start,
        domain_end,
        n_points,
        dtype=s1.control_points.dtype,
        device=s1.control_points.device,
    )

    y1 = b_spline_evaluate(s1, t)
    y2 = b_spline_evaluate(s2, t)
    y_diff = y1 - y2

    from ._b_spline import b_spline_fit

    return b_spline_fit(
        t, y_diff, degree=result_degree, extrapolate=s1.extrapolate
    )


def spline_scale(s: BSpline, c: float) -> BSpline:
    """
    Scale a B-spline by a constant.

    Parameters
    ----------
    s : BSpline
        The B-spline.
    c : float
        Scale factor.

    Returns
    -------
    result : BSpline
        Scaled spline where result(x) = c * s(x).

    Notes
    -----
    This is exact and simply scales the control points.
    """
    return BSpline(
        knots=s.knots.clone(),
        control_points=s.control_points * c,
        degree=s.degree,
        extrapolate=s.extrapolate,
        batch_size=[],
    )


def spline_negate(s: BSpline) -> BSpline:
    """
    Negate a B-spline.

    Parameters
    ----------
    s : BSpline
        The B-spline.

    Returns
    -------
    result : BSpline
        Negated spline where result(x) = -s(x).
    """
    return spline_scale(s, -1.0)


def spline_multiply(s1: BSpline, s2: BSpline) -> BSpline:
    """
    Multiply two B-splines.

    Parameters
    ----------
    s1 : BSpline
        First B-spline.
    s2 : BSpline
        Second B-spline.

    Returns
    -------
    result : BSpline
        Product of the two splines.

    Notes
    -----
    The result spline has degree d1 + d2 where d1 and d2 are the
    degrees of the input splines.

    This uses sampling and re-fitting for simplicity.
    """
    # Determine common domain
    domain_start = max(s1.knots[s1.degree].item(), s2.knots[s2.degree].item())
    domain_end = min(
        s1.knots[-(s1.degree + 1)].item(), s2.knots[-(s2.degree + 1)].item()
    )

    if domain_end <= domain_start:
        raise ValueError("Splines have non-overlapping domains")

    # Product degree
    result_degree = min(s1.degree + s2.degree, 5)  # Cap at degree 5

    # Need more points for higher degree product
    n_points = max(s1.control_points.shape[0], s2.control_points.shape[0]) * (
        result_degree + 1
    )

    t = torch.linspace(
        domain_start,
        domain_end,
        n_points,
        dtype=s1.control_points.dtype,
        device=s1.control_points.device,
    )

    y1 = b_spline_evaluate(s1, t)
    y2 = b_spline_evaluate(s2, t)
    y_prod = y1 * y2

    from ._b_spline import b_spline_fit

    return b_spline_fit(
        t, y_prod, degree=result_degree, extrapolate=s1.extrapolate
    )


def spline_compose(outer: BSpline, inner: BSpline) -> BSpline:
    """
    Compose two B-splines: result(x) = outer(inner(x)).

    Parameters
    ----------
    outer : BSpline
        Outer function.
    inner : BSpline
        Inner function.

    Returns
    -------
    result : BSpline
        Composition outer(inner(x)).

    Notes
    -----
    The inner spline's range must be within the outer spline's domain.
    Uses sampling and re-fitting.
    """
    # Inner spline's domain
    inner_domain_start = inner.knots[inner.degree].item()
    inner_domain_end = inner.knots[-(inner.degree + 1)].item()

    n_points = (
        max(outer.control_points.shape[0], inner.control_points.shape[0]) * 2
    )

    t = torch.linspace(
        inner_domain_start,
        inner_domain_end,
        n_points,
        dtype=inner.control_points.dtype,
        device=inner.control_points.device,
    )

    # Evaluate inner
    y_inner = b_spline_evaluate(inner, t)

    # Check if inner's range is within outer's domain
    outer_domain_start = outer.knots[outer.degree].item()
    outer_domain_end = outer.knots[-(outer.degree + 1)].item()

    # Allow small tolerance for numerical precision
    tol = 1e-6 * (outer_domain_end - outer_domain_start)
    if (
        y_inner.min().item() < outer_domain_start - tol
        or y_inner.max().item() > outer_domain_end + tol
    ):
        raise ValueError(
            f"Inner spline's range [{y_inner.min().item():.4g}, {y_inner.max().item():.4g}] "
            f"exceeds outer spline's domain [{outer_domain_start:.4g}, {outer_domain_end:.4g}]"
        )

    # Clamp inner values to outer domain for robustness
    y_inner = torch.clamp(y_inner, outer_domain_start, outer_domain_end)

    # Evaluate outer at inner's values
    y_composed = b_spline_evaluate(outer, y_inner)

    result_degree = max(outer.degree, inner.degree)

    from ._b_spline import b_spline_fit

    return b_spline_fit(
        t, y_composed, degree=result_degree, extrapolate=outer.extrapolate
    )
