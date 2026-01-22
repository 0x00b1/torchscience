import torch
from torch import Tensor

from ._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
    gegenbauer_polynomial_c,
)


def gegenbauer_polynomial_c_mulx(
    a: GegenbauerPolynomialC,
) -> GegenbauerPolynomialC:
    """Multiply Gegenbauer series by x.

    Uses the recurrence relation:
        x * C_k^{lambda}(x) = ((k+1)/(2*(k+lambda))) * C_{k+1}^{lambda}(x)
                           + ((k+2*lambda-1)/(2*(k+lambda))) * C_{k-1}^{lambda}(x)

    Parameters
    ----------
    a : GegenbauerPolynomialC
        Series to multiply by x.

    Returns
    -------
    GegenbauerPolynomialC
        Series representing x * a(x).

    Notes
    -----
    The degree increases by 1.

    The Gegenbauer recurrence relation is:
        C_{k+1}^{lambda}(x) = (2*(k+lambda)/(k+1))*x*C_k - ((k+2*lambda-1)/(k+1))*C_{k-1}

    Solving for x*C_k(x):
        x*C_k = ((k+1)/(2*(k+lambda)))*C_{k+1} + ((k+2*lambda-1)/(2*(k+lambda)))*C_{k-1}

    Examples
    --------
    >>> a = gegenbauer_polynomial_c(torch.tensor([1.0]), torch.tensor(1.0))  # C_0^1
    >>> b = gegenbauer_polynomial_c_mulx(a)
    >>> b  # x * C_0^1 = x = (1/2)*C_1^1
    GegenbauerPolynomialC(tensor([0.0000, 0.5000]), lambda_=tensor(1.))
    """
    # Get coefficients as plain tensor
    coeffs = a.as_subclass(Tensor)
    lambda_ = a.lambda_
    n = coeffs.shape[-1]

    # Get lambda as scalar for computation
    lambda_val = lambda_.item() if lambda_.dim() == 0 else lambda_[0].item()

    # Result has one more coefficient
    result_shape = list(coeffs.shape)
    result_shape[-1] = n + 1
    result = torch.zeros(
        result_shape, dtype=coeffs.dtype, device=coeffs.device
    )

    # x * C_0^{lambda} = (1/(2*lambda))*C_1^{lambda}
    # From recurrence: C_1 = 2*lambda*x*C_0, so x*C_0 = (1/(2*lambda))*C_1
    result[..., 1] = result[..., 1] + coeffs[..., 0] / (2.0 * lambda_val)

    # x * C_k = ((k+1)/(2*(k+lambda)))*C_{k+1} + ((k+2*lambda-1)/(2*(k+lambda)))*C_{k-1}
    for k in range(1, n):
        c_k = coeffs[..., k]
        denom = 2.0 * (k + lambda_val)

        # Contribution to C_{k-1} coefficient
        coeff_km1 = (k + 2.0 * lambda_val - 1.0) / denom
        result[..., k - 1] = result[..., k - 1] + coeff_km1 * c_k

        # Contribution to C_{k+1} coefficient
        coeff_kp1 = (k + 1.0) / denom
        result[..., k + 1] = result[..., k + 1] + coeff_kp1 * c_k

    return gegenbauer_polynomial_c(result, lambda_)
