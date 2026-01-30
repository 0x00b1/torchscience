"""Beta survival function."""

import torch
from torch import Tensor


def beta_survival(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Survival function (1 - CDF) of the beta distribution.

    .. math::
        S(x; a, b) = 1 - F(x) = I_{1-x}(b, a)

    where :math:`I_z(p, q)` is the regularized incomplete beta function.

    More numerically stable than ``1 - beta_cumulative_distribution(x)`` for
    values of x close to 1.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function. Must be in [0, 1].
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        Survival function values.

    Examples
    --------
    >>> x = torch.tensor([0.25, 0.5, 0.75])
    >>> a = torch.tensor(2.0)
    >>> b = torch.tensor(5.0)
    >>> beta_survival(x, a, b)
    tensor([0.6328, 0.1875, 0.0273])

    See Also
    --------
    beta_cumulative_distribution : CDF = 1 - SF
    """
    return torch.ops.torchscience.beta_survival(x, a, b)
