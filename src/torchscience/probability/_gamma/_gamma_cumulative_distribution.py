"""Gamma cumulative distribution function."""

import torch
from torch import Tensor


def gamma_cumulative_distribution(
    x: Tensor, shape: Tensor, scale: Tensor
) -> Tensor:
    r"""Cumulative distribution function of the gamma distribution.

    .. math::
        F(x; k, \theta) = P(k, x/\theta)

    where :math:`P(a, x)` is the regularized lower incomplete gamma function.

    Parameters
    ----------
    x : Tensor
        Quantiles. Must be non-negative.
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        CDF values.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> shape = torch.tensor(2.0)
    >>> scale = torch.tensor(1.0)
    >>> gamma_cumulative_distribution(x, shape, scale)
    tensor([0.2642, 0.5940, 0.8009])
    """
    return torch.ops.torchscience.gamma_cumulative_distribution(
        x, shape, scale
    )
