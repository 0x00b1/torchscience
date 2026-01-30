"""Gamma survival function."""

import torch
from torch import Tensor


def gamma_survival(x: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    r"""Survival function (1 - CDF) of the gamma distribution.

    .. math::
        S(x; k, \theta) = 1 - F(x) = Q(k, x/\theta)

    where Q is the regularized upper incomplete gamma function.

    More numerically stable than ``1 - gamma_cumulative_distribution(x)`` for large x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function. Must be non-negative.
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        Survival function values.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> shape = torch.tensor(2.0)
    >>> scale = torch.tensor(1.0)
    >>> gamma_survival(x, shape, scale)
    tensor([0.7358, 0.4060, 0.1991])

    See Also
    --------
    gamma_cumulative_distribution : CDF = 1 - SF
    """
    return torch.ops.torchscience.gamma_survival(x, shape, scale)
