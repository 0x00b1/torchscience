import math
import torch
from torch import Tensor

from ._hypergeometric_1_f_1 import hypergeometric_1_f_1


def hypergeometric_u(a: Tensor, b: Tensor, z: Tensor, *, tol: float = 1e-12, max_terms: int = 2048) -> Tensor:
    r"""
    Tricomi confluent hypergeometric function U(a, b, z).

    Uses the connection formula (DLMF 13.1.3) for non-integer b:

    .. math::

       U(a,b,z) = \frac{\pi}{\sin(\pi b)}\left[\frac{\,{}_1F_1(a;b;z)}{\Gamma(1+a-b)\,\Gamma(b)} - z^{1-b}\,\frac{\,{}_1F_1(1+a-b;2-b;z)}{\Gamma(a)\,\Gamma(2-b)}\right].

    Parameters
    ----------
    a : Tensor
        Parameter a. Broadcasting with ``b`` and ``z`` is supported.
    b : Tensor
        Parameter b (singular at integers). For integer b this implementation
        does not attempt the limiting form and returns NaN due to the sin(πb) factor.
    z : Tensor
        Argument (preferably positive real for the power ``z^{1-b}``).
    tol : float, optional
        Relative tolerance passed to the internal 1F1 evaluations.
    max_terms : int, optional
        Maximum terms passed to the internal 1F1 evaluations.

    Returns
    -------
    Tensor
        The value of U(a,b,z) for non-integer b. For b near integers, values may be unstable.
    """
    if not (isinstance(a, Tensor) and isinstance(b, Tensor) and isinstance(z, Tensor)):
        raise TypeError("a, b, z must be torch.Tensors")

    # Promote dtype and broadcast
    if any(t.dtype in (torch.float16, torch.bfloat16) for t in (a, b, z)):
        dtype = torch.float32
    else:
        dtype = z.dtype
    device = z.device

    batch_shape = torch.broadcast_shapes(a.shape, b.shape, z.shape)
    a = a.to(dtype).expand(batch_shape)
    b = b.to(dtype).expand(batch_shape)
    z = z.to(dtype).expand(batch_shape)

    # 1F1 terms
    M1 = hypergeometric_1_f_1(a, b, z, tol=tol, max_terms=max_terms)
    M2 = hypergeometric_1_f_1(1 + a - b, 2 - b, z, tol=tol, max_terms=max_terms)

    # Log-gamma prefactors (real-valued torch.special.gammaln)
    lg_b = torch.special.gammaln(b)
    lg_1ab = torch.special.gammaln(1 + a - b)
    lg_a = torch.special.gammaln(a)
    lg_2mb = torch.special.gammaln(2 - b)

    term1 = torch.exp(-(lg_1ab + lg_b)) * M1
    # z**(1-b) - for z<=0 this is multivalued; tests should avoid that case
    zb = torch.pow(z, 1 - b)
    term2 = torch.exp(-(lg_a + lg_2mb)) * zb * M2

    # Handle sin(pi b)
    sinpib = torch.sin(b * math.pi)
    # Where sinpib==0, return NaN to indicate singular (limit not implemented)
    safe_factor = torch.where(sinpib == 0, torch.nan, math.pi / sinpib)

    return safe_factor * (term1 - term2)


__all__ = ["hypergeometric_u"]

