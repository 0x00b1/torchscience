import math
import torch
from torch import Tensor

from ._hypergeometric_0_f_1 import hypergeometric_0_f_1


def airy_bi(x: Tensor) -> Tensor:
    r"""
    Airy function Bi(x) via 0F1 representations.

    Uses the identities (DLMF 9.4.1–9.4.3):

    Bi(x) = d0 * 0F1(; 2/3; x^3/9) + x * d1 * 0F1(; 4/3; x^3/9)

    where d0 = 1 / (3^{1/6} Γ(2/3)) and d1 = 1 / (3^{5/6} Γ(4/3)).
    """
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    dtype = x.dtype if not x.is_floating_point() else x.dtype
    if dtype in (torch.float16, torch.bfloat16):
        dtype_eff = torch.float32
    else:
        dtype_eff = dtype
    device = x.device

    ln3 = math.log(3.0)
    lg23 = math.lgamma(2.0 / 3.0)
    lg43 = math.lgamma(4.0 / 3.0)
    d0 = math.exp(-(1.0 / 6.0) * ln3 - lg23)
    d1 = math.exp(-(5.0 / 6.0) * ln3 - lg43)
    d0_t = torch.as_tensor(d0, dtype=dtype_eff, device=device)
    d1_t = torch.as_tensor(d1, dtype=dtype_eff, device=device)

    x = x.to(dtype_eff)
    t = (x * x * x) / 9.0

    term0 = d0_t * hypergeometric_0_f_1(torch.as_tensor(2.0 / 3.0, dtype=dtype_eff, device=device), t)
    term1 = x * d1_t * hypergeometric_0_f_1(torch.as_tensor(4.0 / 3.0, dtype=dtype_eff, device=device), t)
    return term0 + term1


__all__ = ["airy_bi"]

