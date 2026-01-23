import torch
from torch import Tensor

from ._hypergeometric_core import _pfq_series


def hypergeometric_0_f_1(b: Tensor, z: Tensor, *, tol: float = 1e-12, max_terms: int = 2048) -> Tensor:
    r"""
    Confluent hypergeometric limit function 0F1(; b; z).

    Computes the generalized hypergeometric function with no numerator parameters
    and one denominator parameter ``b``.

    Series definition
    -----------------
    .. math::

       {}_0F_1(; b; z) = \sum_{k=0}^{\infty} \frac{z^k}{(b)_k\, k!}

    where ``(b)_k`` is the Pochhammer symbol (rising factorial).

    Relation to Bessel I
    --------------------
    For general complex z with principal square root:

    .. math::

       {}_0F_1(; b; z) = \Gamma(b)\, z^{\tfrac{1-b}{2}}\, I_{b-1}(2\sqrt{z}).

    This relation can be used for large ``|z|`` via stable evaluation of
    the modified Bessel function of the first kind, but this implementation
    currently uses the power series with adaptive truncation for stability
    and portability.

    Parameters
    ----------
    b : Tensor
        Denominator parameter. Broadcasting with ``z`` is supported.
    z : Tensor
        Argument tensor. Broadcasting with ``b`` is supported.
    tol : float, optional
        Relative tolerance for adaptive truncation of the series.
    max_terms : int, optional
        Maximum number of series terms.

    Returns
    -------
    Tensor
        The value of ``0F1(; b; z)`` evaluated elementwise.

    Notes
    -----
    - Poles occur when ``b`` is a non-positive integer; values will become
      ``inf`` or ``nan`` accordingly via the series coefficients.
    - Dtype and device follow standard PyTorch promotion rules. Half-precision
      inputs are promoted to float32 internally for accuracy.
    """
    if not isinstance(b, Tensor) or not isinstance(z, Tensor):
        raise TypeError("b and z must be torch.Tensor")
    # Pre-broadcast b and z to a common batch shape, then pass a single
    # denominator parameter with explicit param-dim for stability.
    dtype = z.dtype if z.is_complex() or b.is_complex() else (
        torch.float32 if z.dtype in (torch.float16, torch.bfloat16) or b.dtype in (torch.float16, torch.bfloat16) else z.dtype
    )
    batch_shape = torch.broadcast_shapes(b.shape, z.shape)
    z_b = z.to(dtype).expand(batch_shape)
    b_b = b.to(dtype).expand(batch_shape).unsqueeze(-1)  # append param-dim q=1
    return _pfq_series(a_params=None, b_params=b_b, z=z_b, tol=tol, max_terms=max_terms)


__all__ = ["hypergeometric_0_f_1"]
