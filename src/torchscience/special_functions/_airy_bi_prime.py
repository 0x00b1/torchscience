import math
import torch
from torch import Tensor

from ._hypergeometric_0_f_1 import hypergeometric_0_f_1


def airy_bi_prime(x: Tensor) -> Tensor:
    r"""
    Derivative of Airy function Bi'(x) via 0F1 identities and chain rule.

    Bi(x) = d0 * 0F1(; 2/3; t) + x * d1 * 0F1(; 4/3; t),  t = x^3/9

    With d/dt 0F1(; b; t) = 0F1(; b+1; t) / b and dt/dx = x^2/3.
    """
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    else:
        dtype = x.dtype
    device = x.device

    ln3 = math.log(3.0)
    lg23 = math.lgamma(2.0 / 3.0)
    lg43 = math.lgamma(4.0 / 3.0)
    d0 = math.exp(-(1.0 / 6.0) * ln3 - lg23)
    d1 = math.exp(-(5.0 / 6.0) * ln3 - lg43)
    d0_t = torch.as_tensor(d0, dtype=dtype, device=device)
    d1_t = torch.as_tensor(d1, dtype=dtype, device=device)

    x = x.to(dtype)
    t = (x * x * x) / 9.0
    dt_dx = (x * x) / 3.0

    # First term derivative: d0 * (dt/dx) * 0F1(; 5/3; t) / (2/3)
    term0 = d0_t * dt_dx * hypergeometric_0_f_1(torch.as_tensor(5.0 / 3.0, dtype=dtype, device=device), t) / (
        torch.as_tensor(2.0 / 3.0, dtype=dtype, device=device)
    )

    # Second term derivative: d1 * 0F1(;4/3;t) + x * d1 * (dt/dx) * 0F1(;7/3;t)/(4/3)
    term1a = d1_t * hypergeometric_0_f_1(torch.as_tensor(4.0 / 3.0, dtype=dtype, device=device), t)
    term1b = x * d1_t * dt_dx * hypergeometric_0_f_1(torch.as_tensor(7.0 / 3.0, dtype=dtype, device=device), t) / (
        torch.as_tensor(4.0 / 3.0, dtype=dtype, device=device)
    )

    return term0 + term1a + term1b


__all__ = ["airy_bi_prime"]

