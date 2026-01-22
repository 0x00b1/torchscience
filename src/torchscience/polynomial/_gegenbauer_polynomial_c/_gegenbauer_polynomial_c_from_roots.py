import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
    gegenbauer_polynomial_c,
)
from ._gegenbauer_polynomial_c_multiply import gegenbauer_polynomial_c_multiply


def gegenbauer_polynomial_c_from_roots(
    roots: Tensor,
    lambda_: Tensor,
) -> GegenbauerPolynomialC:
    """Construct monic Gegenbauer series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).
    lambda_ : Tensor
        Parameter lambda > -1/2.

    Returns
    -------
    GegenbauerPolynomialC
        Monic Gegenbauer series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Gegenbauer form.
    Since C_1^{lambda}(x) = 2*lambda*x, we have:
        x = C_1^{lambda}(x) / (2*lambda)

    So (x - r) = (1/(2*lambda))*C_1^{lambda} - r*C_0^{lambda}

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = gegenbauer_polynomial_c_from_roots(roots, torch.tensor(1.0))
    """
    # Ensure lambda_ is a tensor
    if not isinstance(lambda_, Tensor):
        lambda_ = torch.tensor(lambda_, dtype=roots.dtype, device=roots.device)

    n = roots.shape[0]

    if n == 0:
        # Empty roots -> constant 1
        return gegenbauer_polynomial_c(
            torch.ones(1, dtype=roots.dtype, device=roots.device),
            lambda_,
        )

    # For Gegenbauer: C_1^{lambda}(x) = 2*lambda*x
    # So x = C_1^{lambda}(x) / (2*lambda)
    # And (x - r) = -r*C_0 + (1/(2*lambda))*C_1
    lambda_val = lambda_.item() if lambda_.dim() == 0 else lambda_[0].item()

    # Start with (x - r_0)
    result = gegenbauer_polynomial_c(
        torch.tensor(
            [-roots[0], 1.0 / (2.0 * lambda_val)],
            dtype=roots.dtype,
            device=roots.device,
        ),
        lambda_,
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        factor = gegenbauer_polynomial_c(
            torch.tensor(
                [-roots[k], 1.0 / (2.0 * lambda_val)],
                dtype=roots.dtype,
                device=roots.device,
            ),
            lambda_,
        )
        result = gegenbauer_polynomial_c_multiply(result, factor)

    return result
