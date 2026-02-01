"""Generalized hypergeometric function pFq."""

import torch
from torch import Tensor


def hypergeometric_p_f_q(a: Tensor, b: Tensor, z: Tensor) -> Tensor:
    r"""Generalized hypergeometric function :math:`{}_pF_q(a_1,\ldots,a_p;b_1,\ldots,b_q;z)`.

    The generalized hypergeometric function pFq is defined as:

    .. math::
        {}_pF_q(a_1,\ldots,a_p;b_1,\ldots,b_q;z) = \sum_{n=0}^{\infty}
        \frac{\prod_{i=1}^{p}(a_i)_n}{\prod_{j=1}^{q}(b_j)_n} \frac{z^n}{n!}

    where :math:`(x)_n = x(x+1)(x+2)\cdots(x+n-1)` is the Pochhammer symbol
    (rising factorial).

    Convergence:

    - :math:`p \leq q`: converges for all z (entire function)
    - :math:`p = q + 1`: converges for |z| < 1
    - :math:`p > q + 1`: diverges (returns NaN for non-polynomial cases)

    Special cases handled by this implementation:

    - :math:`{}_0F_0(;;z) = e^z`
    - :math:`{}_1F_0(a;;z) = (1-z)^{-a}` for |z| < 1
    - :math:`{}_0F_1(;b;z)` is the confluent hypergeometric limit function
    - :math:`{}_1F_1(a;b;z)` is Kummer's confluent hypergeometric M
    - :math:`{}_2F_1(a,b;c;z)` is the Gauss hypergeometric function

    Parameters
    ----------
    a : Tensor
        Upper parameters. Shape ``[..., p]`` where the last dimension contains
        the p upper parameters and ``...`` are optional batch dimensions.
    b : Tensor
        Lower parameters. Shape ``[..., q]`` where the last dimension contains
        the q lower parameters. Must not contain non-positive integers.
    z : Tensor
        Argument. Shape ``[...]`` representing the batch dimensions.

    Returns
    -------
    Tensor
        The value of :math:`{}_pF_q`. Shape is the broadcast of the batch
        dimensions of ``a``, ``b``, and ``z``.

    Examples
    --------
    Compute :math:`{}_2F_1(1,2;3;0.5)`:

    >>> import torch
    >>> import torchscience.special_functions as sf
    >>> a = torch.tensor([1.0, 2.0])  # Two upper parameters
    >>> b = torch.tensor([3.0])       # One lower parameter
    >>> z = torch.tensor(0.5)
    >>> sf.hypergeometric_p_f_q(a, b, z)
    tensor(...)

    Compute :math:`{}_0F_1(;2;1)` (equivalent to hypergeometric_0_f_1):

    >>> a = torch.tensor([])  # No upper parameters (p=0)
    >>> b = torch.tensor([2.0])
    >>> z = torch.tensor(1.0)
    >>> sf.hypergeometric_p_f_q(a, b, z)
    tensor(...)

    Batched computation:

    >>> a = torch.tensor([[1.0, 2.0], [0.5, 1.5]])  # Shape [2, 2]
    >>> b = torch.tensor([[3.0], [2.0]])            # Shape [2, 1]
    >>> z = torch.tensor([0.5, 0.25])               # Shape [2]
    >>> sf.hypergeometric_p_f_q(a, b, z)
    tensor([..., ...])

    Notes
    -----
    - Returns infinity when any element of ``b`` is a non-positive integer.
    - Returns NaN when the series diverges (p > q + 1 and non-polynomial).
    - When any element of ``a`` is a non-positive integer ``-m``, the series
      terminates and becomes a polynomial of degree ``m``.
    """
    return torch.ops.torchscience.hypergeometric_p_f_q(a, b, z)
