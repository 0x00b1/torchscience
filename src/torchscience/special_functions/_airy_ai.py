import math
import torch
from torch import Tensor

from ._hypergeometric_0_f_1 import hypergeometric_0_f_1


def airy_ai(x: Tensor) -> Tensor:
    r"""
    Airy function Ai(x) via 0F1 representations.

    Uses the identities (DLMF 9.4.1–9.4.3):

    Ai(x) = c0 * 0F1(; 2/3; x^3/9) - x * c1 * 0F1(; 4/3; x^3/9)

    where c0 = 1 / (3^{2/3} Γ(2/3)) and c1 = 1 / (3^{4/3} Γ(4/3)).
    """
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    dtype = x.dtype if not x.is_floating_point() else x.dtype
    if dtype in (torch.float16, torch.bfloat16):
        dtype_eff = torch.float32
    else:
        dtype_eff = dtype
    device = x.device

    # Constants computed in float64 for accuracy, then cast
    ln3 = math.log(3.0)
    lg23 = math.lgamma(2.0 / 3.0)
    lg43 = math.lgamma(4.0 / 3.0)
    c0 = math.exp(-(2.0 / 3.0) * ln3 - lg23)
    c1 = math.exp(-(4.0 / 3.0) * ln3 - lg43)
    c0_t = torch.as_tensor(c0, dtype=dtype_eff, device=device)
    c1_t = torch.as_tensor(c1, dtype=dtype_eff, device=device)

    x = x.to(dtype_eff)
    t = (x * x * x) / 9.0

    term0 = c0_t * hypergeometric_0_f_1(torch.as_tensor(2.0 / 3.0, dtype=dtype_eff, device=device), t)
    term1 = x * c1_t * hypergeometric_0_f_1(torch.as_tensor(4.0 / 3.0, dtype=dtype_eff, device=device), t)
    return term0 - term1


__all__ = ["airy_ai"]

