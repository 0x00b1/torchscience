"""Hypergeometric function 1F2."""

import torch
from torch import Tensor


def hypergeometric_1_f_2(
    a: Tensor, b1: Tensor, b2: Tensor, z: Tensor
) -> Tensor:
    r"""Hypergeometric function :math:`{}_1F_2(a;b_1,b_2;z)`.

    The generalized hypergeometric function 1F2 is defined as:

    .. math::
        {}_1F_2(a;b_1,b_2;z) = \sum_{n=0}^{\infty} \frac{(a)_n}{(b_1)_n (b_2)_n} \frac{z^n}{n!}

    where :math:`(x)_n = x(x+1)(x+2)\cdots(x+n-1)` is the Pochhammer symbol
    (rising factorial).

    This function appears in various contexts including:

    - Solutions to certain differential equations
    - Bessel function integrals
    - Spherical Bessel functions

    Parameters
    ----------
    a : Tensor
        Upper parameter.
    b1 : Tensor
        First lower parameter. Must not be a non-positive integer.
    b2 : Tensor
        Second lower parameter. Must not be a non-positive integer.
    z : Tensor
        Argument.

    Returns
    -------
    Tensor
        The value of :math:`{}_1F_2(a;b_1,b_2;z)`.

    Examples
    --------
    >>> import torch
    >>> import torchscience.special_functions as sf
    >>> a = torch.tensor([1.0])
    >>> b1 = torch.tensor([2.0])
    >>> b2 = torch.tensor([3.0])
    >>> z = torch.tensor([0.5])
    >>> sf.hypergeometric_1_f_2(a, b1, b2, z)
    tensor([1.0455...])

    Notes
    -----
    - Returns infinity when ``b1`` or ``b2`` is a non-positive integer.
    - Returns 1 when ``z = 0``.
    - When ``a`` is a non-positive integer ``-m``, the series terminates
      and becomes a polynomial of degree ``m``.
    """
    return torch.ops.torchscience.hypergeometric_1_f_2(a, b1, b2, z)
