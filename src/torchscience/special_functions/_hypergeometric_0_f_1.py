"""Hypergeometric function 0F1 (confluent hypergeometric limit function)."""

import torch
from torch import Tensor


def hypergeometric_0_f_1(b: Tensor, z: Tensor) -> Tensor:
    r"""Hypergeometric function :math:`{}_0F_1(;b;z)`.

    The confluent hypergeometric limit function is defined as:

    .. math::
        {}_0F_1(;b;z) = \sum_{n=0}^{\infty} \frac{z^n}{(b)_n \cdot n!}

    where :math:`(b)_n = b(b+1)(b+2)\cdots(b+n-1)` is the Pochhammer symbol
    (rising factorial).

    This function is related to Bessel functions:

    .. math::
        J_\nu(z) = \frac{(z/2)^\nu}{\Gamma(\nu+1)} {}_0F_1(;\nu+1;-z^2/4)

    .. math::
        I_\nu(z) = \frac{(z/2)^\nu}{\Gamma(\nu+1)} {}_0F_1(;\nu+1;z^2/4)

    Parameters
    ----------
    b : Tensor
        Parameter. Must not be a non-positive integer.
    z : Tensor
        Argument.

    Returns
    -------
    Tensor
        The value of :math:`{}_0F_1(;b;z)`.

    Examples
    --------
    >>> import torch
    >>> import torchscience.special_functions as sf
    >>> b = torch.tensor([2.0])
    >>> z = torch.tensor([1.0])
    >>> sf.hypergeometric_0_f_1(b, z)
    tensor([1.2660...])

    Notes
    -----
    - Returns infinity when ``b`` is a non-positive integer.
    - Returns 1 when ``z = 0``.
    """
    return torch.ops.torchscience.hypergeometric_0_f_1(b, z)
