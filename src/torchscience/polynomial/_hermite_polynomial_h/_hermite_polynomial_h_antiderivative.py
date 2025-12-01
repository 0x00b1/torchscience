import torch

from ._hermite_polynomial_h import HermitePolynomialH
from ._hermite_polynomial_h_evaluate import hermite_polynomial_h_evaluate


def hermite_polynomial_h_antiderivative(
    a: HermitePolynomialH,
    order: int = 1,
    constant: float = 0.0,
) -> HermitePolynomialH:
    """Compute antiderivative of Physicists' Hermite series.

    Uses the integration formula for Hermite polynomials:
        integral(H_n(x)) dx = H_{n+1}(x) / (2*(n+1))

    The constant of integration is chosen such that the antiderivative
    evaluates to `constant` at x=0.

    Parameters
    ----------
    a : HermitePolynomialH
        Series to integrate.
    order : int, optional
        Order of integration. Default is 1.
    constant : float, optional
        Integration constant. The antiderivative will evaluate to this
        value at x=0. Default is 0.0.

    Returns
    -------
    HermitePolynomialH
        Antiderivative series.

    Notes
    -----
    The degree increases by 1 for each integration.

    For Hermite polynomials:
        d/dx H_n(x) = 2n * H_{n-1}(x)

    Therefore:
        integral(H_n(x)) dx = H_{n+1}(x) / (2*(n+1)) + C

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0]))  # H_0 = 1
    >>> ia = hermite_polynomial_h_antiderivative(a)
    >>> ia.coeffs  # integral(1) = x = H_1/2
    tensor([0., 0.5])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return HermitePolynomialH(coeffs=a.coeffs.clone())

    coeffs = a.coeffs

    # Apply antiderivative 'order' times
    for i in range(order):
        n = coeffs.shape[-1]

        # Result has n+1 coefficients (degree increases by 1)
        result_shape = list(coeffs.shape)
        result_shape[-1] = n + 1
        i_coeffs = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # integral(H_k) dx = H_{k+1} / (2*(k+1))
        # So coefficient of H_{k+1} in integral is c_k / (2*(k+1))
        for k in range(n):
            i_coeffs[..., k + 1] = coeffs[..., k] / (2.0 * (k + 1))

        # Set constant of integration so F(0) = constant (for first integration)
        # or F(0) = 0 (for subsequent integrations)
        k_val = constant if i == 0 else 0.0
        temp = HermitePolynomialH(coeffs=i_coeffs)
        x_zero = torch.zeros((), dtype=coeffs.dtype, device=coeffs.device)
        val_at_zero = hermite_polynomial_h_evaluate(temp, x_zero)
        # H_0(x) = 1, so adding delta to i_coeffs[..., 0] shifts F(0) by delta
        i_coeffs[..., 0] = i_coeffs[..., 0] + (k_val - val_at_zero)

        coeffs = i_coeffs

    return HermitePolynomialH(coeffs=coeffs)
