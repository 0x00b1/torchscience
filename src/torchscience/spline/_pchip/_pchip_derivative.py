"""PCHIP spline derivative computation."""

import torch

from ._pchip import PCHIPSpline


def pchip_derivative(
    spline: PCHIPSpline,
    order: int = 1,
) -> PCHIPSpline:
    """
    Compute the derivative of a PCHIP spline.

    Parameters
    ----------
    spline : PCHIPSpline
        Input PCHIP spline
    order : int
        Order of derivative (1, 2, or 3). Default is 1.

    Returns
    -------
    derivative : PCHIPSpline
        A new PCHIPSpline representing the derivative.
        First derivative is quadratic (degree 2), second is linear (degree 1),
        third is constant (degree 0).

    Raises
    ------
    ValueError
        If order is not 1, 2, or 3.

    Notes
    -----
    For cubic polynomial: y = a + b*dx + c*dx^2 + d*dx^3
    - First derivative: y' = b + 2c*dx + 3d*dx^2  (coefficients: [b, 2c, 3d, 0])
    - Second derivative: y'' = 2c + 6d*dx  (coefficients: [2c, 6d, 0, 0])
    - Third derivative: y''' = 6d  (coefficients: [6d, 0, 0, 0])

    The new PCHIPSpline has the same knots but transformed coefficients.
    """
    if order < 1 or order > 3:
        raise ValueError(f"Derivative order must be 1, 2, or 3, got {order}")

    coeffs = spline.coefficients

    # Extract coefficients
    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    d = coeffs[:, 3]

    if order == 1:
        # First derivative: b + 2c*dx + 3d*dx^2
        new_a = b
        new_b = 2 * c
        new_c = 3 * d
        new_d = torch.zeros_like(d)
    elif order == 2:
        # Second derivative: 2c + 6d*dx
        new_a = 2 * c
        new_b = 6 * d
        new_c = torch.zeros_like(c)
        new_d = torch.zeros_like(d)
    else:  # order == 3
        # Third derivative: 6d
        new_a = 6 * d
        new_b = torch.zeros_like(b)
        new_c = torch.zeros_like(c)
        new_d = torch.zeros_like(d)

    new_coeffs = torch.stack([new_a, new_b, new_c, new_d], dim=1)

    return PCHIPSpline(
        knots=spline.knots.clone(),
        coefficients=new_coeffs,
        extrapolate=spline.extrapolate,
        batch_size=[],
    )
