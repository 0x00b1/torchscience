import torch
from torch import Tensor

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)
from ._chebyshev_polynomial_u_multiply import chebyshev_polynomial_u_multiply


def chebyshev_polynomial_u_from_roots(
    roots: Tensor,
) -> ChebyshevPolynomialU:
    """Construct monic Chebyshev U series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    ChebyshevPolynomialU
        Monic Chebyshev U series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Chebyshev U form.
    Since x = U_1/2 and constants are U_0, (x - r) = -r*U_0 + U_1/2.

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = chebyshev_polynomial_u_from_roots(roots)
    """
    n = roots.shape[0]

    if n == 0:
        # Empty roots -> constant 1
        return chebyshev_polynomial_u(
            torch.ones(1, dtype=roots.dtype, device=roots.device)
        )

    # Start with (x - r_0) = -r_0 * U_0 + (1/2) * U_1
    # Since x = U_1/2 in U-basis
    result = chebyshev_polynomial_u(
        torch.tensor([-roots[0], 0.5], dtype=roots.dtype, device=roots.device)
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        factor = chebyshev_polynomial_u(
            torch.tensor(
                [-roots[k], 0.5], dtype=roots.dtype, device=roots.device
            )
        )
        result = chebyshev_polynomial_u_multiply(result, factor)

    return result
