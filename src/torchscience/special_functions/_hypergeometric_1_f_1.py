import torch
from torch import Tensor

from ._hypergeometric_core import _pfq_series


def hypergeometric_1_f_1(a: Tensor, b: Tensor, z: Tensor, *, tol: float = 1e-12, max_terms: int = 2048) -> Tensor:
    r"""
    Confluent hypergeometric function 1F1(a; b; z) (Kummer M).

    Series definition
    -----------------
    .. math::

       {}_1F_1(a; b; z) = \sum_{k=0}^{\infty} \frac{(a)_k}{(b)_k} \frac{z^k}{k!}

    Parameters
    ----------
    a : Tensor
        Numerator parameter. Broadcasting with ``b`` and ``z`` is supported.
    b : Tensor
        Denominator parameter (pole when ``b`` is a non-positive integer).
    z : Tensor
        Argument tensor.
    tol : float, optional
        Relative tolerance for adaptive truncation of the series.
    max_terms : int, optional
        Maximum number of series terms.

    Returns
    -------
    Tensor
        The value of ``1F1(a; b; z)``.
    """
    if not (isinstance(a, Tensor) and isinstance(b, Tensor) and isinstance(z, Tensor)):
        raise TypeError("a, b, z must be torch.Tensors")
    # Pre-broadcast to a common batch shape, then create explicit param dims
    # of length 1 for a and b
    # Dtype promotion: at least float32 for half types
    if any(t.dtype in (torch.float16, torch.bfloat16) for t in (a, b, z)):
        dtype = torch.float32
    else:
        dtype = z.dtype
    batch_shape = torch.broadcast_shapes(a.shape, b.shape, z.shape)
    a_b = a.to(dtype).expand(batch_shape).unsqueeze(-1)
    b_b = b.to(dtype).expand(batch_shape).unsqueeze(-1)
    z_b = z.to(dtype).expand(batch_shape)
    return _pfq_series(a_params=a_b, b_params=b_b, z=z_b, tol=tol, max_terms=max_terms)


__all__ = ["hypergeometric_1_f_1"]

