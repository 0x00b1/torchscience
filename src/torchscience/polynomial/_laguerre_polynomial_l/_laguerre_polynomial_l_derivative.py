import torch

from ._laguerre_polynomial_l import LaguerrePolynomialL


def laguerre_polynomial_l_derivative(
    a: LaguerrePolynomialL,
    order: int = 1,
) -> LaguerrePolynomialL:
    """Compute derivative of Laguerre series.

    Uses the recurrence relation:
        d/dx L_n(x) = -sum_{k=0}^{n-1} L_k(x)

    which leads to the algorithm (matching numpy's lagder):
        der[j] = -sum_{k=j+1}^{n-1} c[k]  for j = 0, ..., n-2

    Parameters
    ----------
    a : LaguerrePolynomialL
        Series to differentiate.
    order : int, optional
        Order of derivative. Default is 1.

    Returns
    -------
    LaguerrePolynomialL
        Derivative series.

    Notes
    -----
    The degree decreases by 1 for each differentiation.

    The derivative of L_n is:
        d/dx L_n(x) = -L_0(x) - L_1(x) - ... - L_{n-1}(x)

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([0.0, 1.0]))  # L_1
    >>> da = laguerre_polynomial_l_derivative(a)
    >>> da.coeffs  # d/dx L_1 = -L_0
    tensor([-1.])
    """
    if order < 0:
        raise ValueError(f"Order must be non-negative, got {order}")

    if order == 0:
        return LaguerrePolynomialL(coeffs=a.coeffs.clone())

    coeffs = a.coeffs.clone()  # Will be modified during iteration
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

        # For Laguerre: d/dx L_k = -sum_{j=0}^{k-1} L_j
        # So for f = sum c_k L_k, the derivative coefficient for L_j is:
        # der[j] = -sum_{k=j+1}^{n-1} c[k]
        #
        # This can be computed as a cumulative sum from the end:
        # der[n-2] = -c[n-1]
        # der[n-3] = -c[n-1] - c[n-2] = der[n-2] - c[n-2]
        # ...
        # der[j] = der[j+1] - c[j+1]

        cumsum = torch.zeros_like(coeffs[..., 0])
        for j in range(new_n - 1, -1, -1):
            cumsum = cumsum + coeffs[..., j + 1]
            der[..., j] = -cumsum

        coeffs = der
        n = new_n

    return LaguerrePolynomialL(coeffs=coeffs)
