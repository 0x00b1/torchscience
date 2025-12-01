import torch

from torchscience.polynomial._exceptions import ParameterMismatchError

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC


def gegenbauer_polynomial_c_subtract(
    a: GegenbauerPolynomialC,
    b: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Subtract two Gegenbauer series.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        First series.
    b : GegenbauerPolynomialC
        Second series.

    Returns
    -------
    GegenbauerPolynomialC
        Difference a - b.

    Raises
    ------
    ParameterMismatchError
        If the series have different lambda parameters.

    Notes
    -----
    If the series have different degrees, the shorter one is zero-padded.
    Both series must have the same lambda parameter (within numerical tolerance).

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([5.0, 7.0, 9.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(1.0))
    >>> c = gegenbauer_polynomial_c_subtract(a, b)
    >>> c.coeffs
    tensor([4., 5., 6.])
    """
    # Check parameter compatibility
    if not torch.allclose(a.lambda_, b.lambda_):
        raise ParameterMismatchError(
            f"Cannot subtract GegenbauerPolynomialC with lambda={b.lambda_} "
            f"from GegenbauerPolynomialC with lambda={a.lambda_}"
        )

    a_coeffs = a.coeffs
    b_coeffs = b.coeffs

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a == n_b:
        return GegenbauerPolynomialC(
            coeffs=a_coeffs - b_coeffs, lambda_=a.lambda_
        )

    # Zero-pad the shorter series
    if n_a < n_b:
        pad_shape = list(a_coeffs.shape)
        pad_shape[-1] = n_b - n_a
        padding = torch.zeros(
            pad_shape, dtype=a_coeffs.dtype, device=a_coeffs.device
        )
        a_coeffs = torch.cat([a_coeffs, padding], dim=-1)
    else:
        pad_shape = list(b_coeffs.shape)
        pad_shape[-1] = n_a - n_b
        padding = torch.zeros(
            pad_shape, dtype=b_coeffs.dtype, device=b_coeffs.device
        )
        b_coeffs = torch.cat([b_coeffs, padding], dim=-1)

    return GegenbauerPolynomialC(coeffs=a_coeffs - b_coeffs, lambda_=a.lambda_)
