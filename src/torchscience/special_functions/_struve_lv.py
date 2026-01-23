import math
import torch
from torch import Tensor

from ._hypergeometric_core import _pfq_series


def struve_lv(nu: Tensor, z: Tensor, *, tol: float = 1e-12, max_terms: int = 4096) -> Tensor:
    r"""
    Modified Struve function L_ν(z) via 1F2 representation.

    Uses (DLMF 11.2.6):

    .. math::

       L_{\nu}(z) = \frac{(z/2)^{\nu+1}}{\sqrt{\pi}\, \Gamma(\nu+3/2)}\, {}_1F_2\!\left(1;\,\tfrac{3}{2},\,\nu+\tfrac{3}{2};\, +\tfrac{z^2}{4}\right).

    Parameters
    ----------
    nu : Tensor
        Order ν (real). Broadcasting with ``z`` is supported.
    z : Tensor
        Argument (real). Broadcasting with ``nu`` is supported.
    tol : float
        Relative tolerance for series truncation.
    max_terms : int
        Maximum number of series terms.

    Returns
    -------
    Tensor
        L_ν(z) evaluated elementwise.
    """
    if not (isinstance(nu, Tensor) and isinstance(z, Tensor)):
        raise TypeError("nu and z must be torch.Tensors")

    # Promote dtype
    if any(t.dtype in (torch.float16, torch.bfloat16) for t in (nu, z)):
        dtype = torch.float32
    else:
        dtype = z.dtype
    device = z.device

    # Broadcast
    batch_shape = torch.broadcast_shapes(nu.shape, z.shape)
    nu = nu.to(dtype).expand(batch_shape)
    z = z.to(dtype).expand(batch_shape)

    # 1F2 parameters: a = 1; b1 = 3/2; b2 = nu+3/2
    a = torch.ones((*batch_shape, 1), dtype=dtype, device=device)
    b1 = torch.full((*batch_shape, 1), 1.5, dtype=dtype, device=device)
    b2 = (nu + 1.5).unsqueeze(-1)
    b_params = torch.cat([b1, b2], dim=-1)

    t = (z * z) / 4.0
    F = _pfq_series(a_params=a, b_params=b_params, z=t, tol=tol, max_terms=max_terms)

    # Prefactor: 2 * (z/2)^{nu+1} / (sqrt(pi) * Gamma(nu+3/2))
    lg_denom = 0.5 * math.log(math.pi) + torch.special.gammaln(nu + 1.5)
    pref = (2.0 * torch.exp(-lg_denom)) * torch.pow(z / 2.0, nu + 1.0)

    return pref * F


__all__ = ["struve_lv"]
