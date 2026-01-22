import torch
from torch import Tensor

from ._chebyshev_polynomial_w import chebyshev_polynomial_w
from ._chebyshev_polynomial_w_multiply import chebyshev_polynomial_w_multiply


def chebyshev_polynomial_w_from_roots(
    roots: Tensor,
) -> "chebyshev_polynomial_w":
    """Construct monic Chebyshev W series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).

    Returns
    -------
    ChebyshevPolynomialW
        Monic Chebyshev W series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Chebyshev W form.
    For Chebyshev W: W_0 = 1, W_1 = 2x + 1
    So x = (W_1 - 1)/2 = 0.5*W_1 - 0.5
    Thus (x - r) = 0.5*W_1 - 0.5 - r = (-0.5 - r)*W_0 + 0.5*W_1

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = chebyshev_polynomial_w_from_roots(roots)
    """
    n = roots.shape[0]

    if n == 0:
        # Empty roots -> constant 1
        return chebyshev_polynomial_w(
            torch.ones(1, dtype=roots.dtype, device=roots.device)
        )

    # For Chebyshev W: x = (W_1 - 1)/2 = -0.5 + 0.5*W_1
    # So (x - r) = -0.5 - r + 0.5*W_1 = (-0.5 - r)*W_0 + 0.5*W_1
    # Since W_0 = 1, we have coefficients [(-0.5 - r), 0.5]
    result = chebyshev_polynomial_w(
        torch.tensor(
            [-0.5 - roots[0], 0.5], dtype=roots.dtype, device=roots.device
        )
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        factor = chebyshev_polynomial_w(
            torch.tensor(
                [-0.5 - roots[k], 0.5], dtype=roots.dtype, device=roots.device
            )
        )
        result = chebyshev_polynomial_w_multiply(result, factor)

    return result
