import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p
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
        Jacobi parameter alpha, must be > -1.
    beta : Tensor or float
        Jacobi parameter beta, must be > -1.

    Returns
    -------
    JacobiPolynomialP
        Monic Jacobi series with the given roots.

    Notes
    -----
    Builds the product of linear factors (x - r_k) in Jacobi form.
    For Jacobi polynomials with parameters (alpha, beta), the linear factor
    (x - r) must be expressed in the Jacobi basis.

    Since x = (2/(alpha+beta+2)) * P_1^{(alpha,beta)} + ((alpha-beta)/(alpha+beta+2)) * P_0^{(alpha,beta)},
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
        return jacobi_polynomial_p(
            torch.ones(1, dtype=roots.dtype, device=roots.device),
            alpha,
            beta,
        )

    # For Jacobi basis:
    # P_0^{(alpha,beta)}(x) = 1
    # P_1^{(alpha,beta)}(x) = (alpha-beta)/2 + (alpha+beta+2)/2 * x
    #
    # So x = (2*P_1 - (alpha-beta)*P_0) / (alpha+beta+2)
    # And (x - r) = (2*P_1 - (alpha-beta)*P_0 - r*(alpha+beta+2)*P_0) / (alpha+beta+2)
    #             = (2*P_1 - ((alpha-beta) + r*(alpha+beta+2))*P_0) / (alpha+beta+2)

    # Build (x - r_0) in Jacobi form
    denom = ab + 2
    # (x - r) = c_0 * P_0 + c_1 * P_1
    # where c_1 = 2 / (alpha+beta+2) and c_0 = -((alpha-beta) + r*(alpha+beta+2)) / (alpha+beta+2)
    c_0 = -((alpha - beta) + roots[0] * denom) / denom
    c_1 = 2.0 / denom

    result = jacobi_polynomial_p(
        torch.tensor([c_0, c_1], dtype=roots.dtype, device=roots.device),
        alpha.clone(),
        beta.clone(),
    )

    # Multiply by each subsequent (x - r_k)
    for k in range(1, n):
        c_0 = -((alpha - beta) + roots[k] * denom) / denom
        factor = jacobi_polynomial_p(
            torch.tensor([c_0, c_1], dtype=roots.dtype, device=roots.device),
            alpha.clone(),
            beta.clone(),
        )
        result = jacobi_polynomial_p_multiply(result, factor)

    return result
