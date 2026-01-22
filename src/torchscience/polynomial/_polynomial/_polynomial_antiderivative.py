from typing import Union

import torch
from torch import Tensor

from ._polynomial import Polynomial, polynomial


def polynomial_antiderivative(
    p: Polynomial,
    constant: Union[Tensor, float] = 0.0,
) -> Polynomial:
    """Compute antiderivative (indefinite integral).

    Parameters
    ----------
    p : Polynomial
        Input polynomial.
    constant : Tensor or float
        Integration constant (default 0).

    Returns
    -------
    Polynomial
        Antiderivative with given constant term. Degree increases by 1.

    Examples
    --------
    >>> p = polynomial(torch.tensor([2.0, 6.0]))  # 2 + 6x
    >>> polynomial_antiderivative(p)  # 0 + 2x + 3x^2
    Polynomial(tensor([0., 2., 3.]))
    """
    # p IS the coefficient tensor now
    n = p.shape[-1]

    # Integral of (a_0 + a_1*x + ... + a_n*x^n)
    # = C + a_0*x + a_1*x^2/2 + a_2*x^3/3 + ... + a_n*x^(n+1)/(n+1)
    # new_coeffs[0] = constant
    # new_coeffs[i+1] = old_coeffs[i] / (i+1)

    indices = torch.arange(1, n + 1, device=p.device, dtype=p.dtype)
    integrated = p / indices

    # Handle constant term
    if isinstance(constant, Tensor):
        c = constant
    else:
        c = torch.tensor(constant, dtype=p.dtype, device=p.device)

    # Expand constant to match batch dimensions
    if p.dim() > 1 and c.dim() == 0:
        c = c.expand(*p.shape[:-1])

    if c.dim() == 0:
        c = c.unsqueeze(-1)
    else:
        c = c.unsqueeze(-1)

    new_coeffs = torch.cat([c, integrated], dim=-1)
    return polynomial(new_coeffs)
