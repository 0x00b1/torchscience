import torch
from torch import Tensor

from torchscience.polynomial._exceptions import ParameterMismatchError

from ._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
    gegenbauer_polynomial_c,
)


def gegenbauer_polynomial_c_add(
    a: GegenbauerPolynomialC,
    b: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Add two Gegenbauer series.

    Parameters
    ----------
    a : GegenbauerPolynomialC
        First series.
    b : GegenbauerPolynomialC
        Second series.

    Returns
    -------
    GegenbauerPolynomialC
        Sum a + b.

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
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0, 2.0]), torch.tensor(1.0))
    >>> b = gegenbauer_polynomial_c(torch.tensor([3.0, 4.0, 5.0]), torch.tensor(1.0))
    >>> c = gegenbauer_polynomial_c_add(a, b)
    >>> c
    GegenbauerPolynomialC(tensor([4., 6., 5.]), lambda_=tensor(1.))
    """
    # Check parameter compatibility
    if not torch.allclose(a.lambda_, b.lambda_):
        raise ParameterMismatchError(
            f"Cannot add GegenbauerPolynomialC with lambda={a.lambda_} "
            f"to GegenbauerPolynomialC with lambda={b.lambda_}"
        )

    # Get coefficients as plain tensors
    a_coeffs = a.as_subclass(Tensor)
    b_coeffs = b.as_subclass(Tensor)

    n_a = a_coeffs.shape[-1]
    n_b = b_coeffs.shape[-1]

    if n_a == n_b:
        return gegenbauer_polynomial_c(a_coeffs + b_coeffs, a.lambda_)

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

    return gegenbauer_polynomial_c(a_coeffs + b_coeffs, a.lambda_)
