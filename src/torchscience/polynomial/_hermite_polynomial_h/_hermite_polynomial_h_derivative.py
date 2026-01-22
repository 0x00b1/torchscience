import torch

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)


def hermite_polynomial_h_derivative(
    a: HermitePolynomialH,
    order: int = 1,
) -> HermitePolynomialH:
    """Compute derivative of Physicists' Hermite series.

    Uses the derivative formula for Hermite polynomials:
        d/dx H_n(x) = 2n * H_{n-1}(x)

    Parameters
    ----------
    a : HermitePolynomialH
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    HermitePolynomialH
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    For a Hermite series f(x) = sum_{k=0}^{n} c_k H_k(x),
    the derivative is:
        f'(x) = sum_{k=0}^{n} c_k * 2k * H_{k-1}(x)
              = sum_{j=0}^{n-1} (2*(j+1) * c_{j+1}) * H_j(x)

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([0.0, 1.0]))  # H_1 = 2x
    >>> da = hermite_polynomial_h_derivative(a)
    >>> da  # d/dx H_1 = 2*1*H_0 = 2
    HermitePolynomialH(tensor([2.]))
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return hermite_polynomial_h(a.clone())

    coeffs = a.as_subclass(torch.Tensor).clone()
    n = coeffs.shape[-1]

    # Apply derivative 'order' times
    for _ in range(order):
        if n <= 1:
            # Derivative of constant is zero
            result_shape = list(coeffs.shape)
            result_shape[-1] = 1
            coeffs = torch.zeros(
                result_shape, dtype=coeffs.dtype, device=coeffs.device
            )
            n = 1
            continue

        # Result has n-1 coefficients (degree decreases by 1)
        new_n = n - 1
        result_shape = list(coeffs.shape)
        result_shape[-1] = new_n
        der = torch.zeros(
            result_shape, dtype=coeffs.dtype, device=coeffs.device
        )

        # d/dx H_k = 2k * H_{k-1}
        # So coefficient of H_j in derivative is 2*(j+1) * c_{j+1}
        for j in range(new_n):
            der[..., j] = 2.0 * (j + 1) * coeffs[..., j + 1]

        coeffs = der
        n = new_n

    return hermite_polynomial_h(coeffs)
