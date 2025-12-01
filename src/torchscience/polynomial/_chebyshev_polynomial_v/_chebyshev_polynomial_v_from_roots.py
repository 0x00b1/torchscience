import torch
from torch import Tensor

from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_multiply import chebyshev_polynomial_v_multiply


def chebyshev_polynomial_v_from_roots(
    roots: Tensor,
) -> ChebyshevPolynomialV:
    """Construct monic Chebyshev V series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    ChebyshevPolynomialV
        Monic Chebyshev V series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Chebyshev V form.
    Since x = (V_0 + V_1)/2 in Chebyshev V basis and constants map to V_0,
    (x - r) needs to be expressed in V basis.

    For Chebyshev V: x = (V_0 + V_1 + 1)/2 = 0.5*V_0 + 0.5*V_1 + 0.5
    So (x - r) = (0.5 - r)*V_0 + 0.5*V_1

    Actually, from V_1 = 2x - 1, we get x = (V_1 + 1)/2
    So (x - r) = -r + (V_1 + 1)/2 = (0.5 - r) + 0.5*V_1
              = (0.5 - r)*V_0 + 0.5*V_1

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = chebyshev_polynomial_v_from_roots(roots)
    """
    n = roots.shape[0]

    if n == 0:
        # Empty roots -> constant 1
        return ChebyshevPolynomialV(
            coeffs=torch.ones(1, dtype=roots.dtype, device=roots.device)
        )

    # For Chebyshev V: x = (V_1 + 1)/2 = 0.5 + 0.5*V_1
    # So (x - r) = 0.5 - r + 0.5*V_1 = (0.5 - r)*V_0 + 0.5*V_1
    # Since V_0 = 1, we have coefficients [(0.5 - r), 0.5]
    result = ChebyshevPolynomialV(
        coeffs=torch.tensor(
            [0.5 - roots[0], 0.5], dtype=roots.dtype, device=roots.device
        )
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        factor = ChebyshevPolynomialV(
            coeffs=torch.tensor(
                [0.5 - roots[k], 0.5], dtype=roots.dtype, device=roots.device
            )
        )
        result = chebyshev_polynomial_v_multiply(result, factor)

    return result
