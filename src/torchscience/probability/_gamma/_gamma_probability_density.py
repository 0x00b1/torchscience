"""Gamma probability density function."""

import torch
from torch import Tensor


def gamma_probability_density(
    x: Tensor, shape: Tensor, scale: Tensor
) -> Tensor:
    r"""Probability density function of the gamma distribution.

    .. math::
        f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}

    Parameters
    ----------
    x : Tensor
        Values. Must be non-negative.
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        PDF values.
    """
    return torch.ops.torchscience.gamma_probability_density(x, shape, scale)
