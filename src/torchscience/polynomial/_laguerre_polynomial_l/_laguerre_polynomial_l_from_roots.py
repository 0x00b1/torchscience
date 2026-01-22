import torch
from torch import Tensor

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
    laguerre_polynomial_l,
)
from ._laguerre_polynomial_l_multiply import laguerre_polynomial_l_multiply


def laguerre_polynomial_l_from_roots(
    roots: Tensor,
) -> LaguerrePolynomialL:
    """Construct monic Laguerre series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    LaguerrePolynomialL
        Monic Laguerre series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Laguerre form.

    In Laguerre basis:
        x = L_0 - L_1 + L_0 = 2*L_0 - L_1  (since L_0 = 1 and L_1 = 1-x)

    Actually: x = 1 - L_1 = L_0 - L_1 (since L_0 = 1)

    So (x - r) = (1 - L_1) - r = (1 - r) - L_1 = (1-r)*L_0 - L_1

    Examples
    --------
    >>> roots = torch.tensor([1.0, 2.0])
    >>> c = laguerre_polynomial_l_from_roots(roots)
    """
    n = roots.shape[0]

    if n == 0:
        # Empty roots -> constant 1
        return laguerre_polynomial_l(
            torch.ones(1, dtype=roots.dtype, device=roots.device)
        )

    # In Laguerre basis: x = L_0 - L_1 (since L_0 = 1 and L_1 = 1 - x)
    # So (x - r) = (1 - r)*L_0 - L_1

    # Start with (x - r_0) = (1 - r_0)*L_0 - L_1
    result = laguerre_polynomial_l(
        torch.tensor(
            [1.0 - roots[0], -1.0], dtype=roots.dtype, device=roots.device
        )
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        factor = laguerre_polynomial_l(
            torch.tensor(
                [1.0 - roots[k], -1.0], dtype=roots.dtype, device=roots.device
            )
        )
        result = laguerre_polynomial_l_multiply(result, factor)

    return result
