import torch
from torch import Tensor

from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_multiply import legendre_polynomial_p_multiply


def legendre_polynomial_p_from_roots(
    roots: Tensor,
) -> LegendrePolynomialP:
    """Construct monic Legendre series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    LegendrePolynomialP
        Monic Legendre series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Legendre form.
    Since x = P_1 and constants scale P_0, (x - r) = -r*P_0 + P_1.

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = legendre_polynomial_p_from_roots(roots)
    """
    n = roots.shape[0]

    if n == 0:
        # Empty roots -> constant 1
        return LegendrePolynomialP(
            coeffs=torch.ones(1, dtype=roots.dtype, device=roots.device)
        )

    # Start with (x - r_0) = -r_0 * P_0 + P_1
    result = LegendrePolynomialP(
        coeffs=torch.tensor(
            [-roots[0], 1.0], dtype=roots.dtype, device=roots.device
        )
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        factor = LegendrePolynomialP(
            coeffs=torch.tensor(
                [-roots[k], 1.0], dtype=roots.dtype, device=roots.device
            )
        )
        result = legendre_polynomial_p_multiply(result, factor)

    return result
