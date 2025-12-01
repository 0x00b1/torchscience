import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_multiply import jacobi_polynomial_p_multiply


def jacobi_polynomial_p_from_roots(
    roots: Tensor,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> JacobiPolynomialP:
    """Construct monic Jacobi series from its roots.

    The resulting series is (x - r_0)(x - r_1)...(x - r_{n-1}).

    Parameters
    ----------
    roots : Tensor
        Roots of the polynomial, shape (n,).
    alpha : Tensor or float
        Jacobi parameter α, must be > -1.
    beta : Tensor or float
        Jacobi parameter β, must be > -1.

    Returns
    -------
    JacobiPolynomialP
        Monic Jacobi series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Jacobi form.
    For Jacobi polynomials with parameters (α, β), the linear factor
    (x - r) must be expressed in the Jacobi basis.

    Since x = (2/(α+β+2)) * P_1^{(α,β)} + ((α-β)/(α+β+2)) * P_0^{(α,β)},
    the factor (x - r) can be written in terms of P_0 and P_1.

    Examples
    --------
    >>> roots = torch.tensor([0.5, -0.5])
    >>> c = jacobi_polynomial_p_from_roots(roots, alpha=0.0, beta=0.0)
    """
    n = roots.shape[0]

    # Convert alpha and beta to tensors if needed
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(alpha, dtype=roots.dtype, device=roots.device)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=roots.dtype, device=roots.device)

    ab = alpha + beta

    if n == 0:
        # Empty roots -> constant 1
        return JacobiPolynomialP(
            coeffs=torch.ones(1, dtype=roots.dtype, device=roots.device),
            alpha=alpha,
            beta=beta,
        )

    # For Jacobi basis:
    # P_0^{(α,β)}(x) = 1
    # P_1^{(α,β)}(x) = (α-β)/2 + (α+β+2)/2 * x
    #
    # So x = (2*P_1 - (α-β)*P_0) / (α+β+2)
    # And (x - r) = (2*P_1 - (α-β)*P_0 - r*(α+β+2)*P_0) / (α+β+2)
    #             = (2*P_1 - ((α-β) + r*(α+β+2))*P_0) / (α+β+2)

    # Build (x - r_0) in Jacobi form
    denom = ab + 2
    # (x - r) = c_0 * P_0 + c_1 * P_1
    # where c_1 = 2 / (α+β+2) and c_0 = -((α-β) + r*(α+β+2)) / (α+β+2)
    c_0 = -((alpha - beta) + roots[0] * denom) / denom
    c_1 = 2.0 / denom

    result = JacobiPolynomialP(
        coeffs=torch.tensor(
            [c_0, c_1], dtype=roots.dtype, device=roots.device
        ),
        alpha=alpha.clone(),
        beta=beta.clone(),
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        c_0 = -((alpha - beta) + roots[k] * denom) / denom
        factor = JacobiPolynomialP(
            coeffs=torch.tensor(
                [c_0, c_1], dtype=roots.dtype, device=roots.device
            ),
            alpha=alpha.clone(),
            beta=beta.clone(),
        )
        result = jacobi_polynomial_p_multiply(result, factor)

    return result
