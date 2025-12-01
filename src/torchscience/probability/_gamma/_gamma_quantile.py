"""Gamma quantile function."""

import torch
from torch import Tensor


def gamma_quantile(p: Tensor, shape: Tensor, scale: Tensor) -> Tensor:
    r"""Quantile function (inverse CDF) of the gamma distribution.

    Parameters
    ----------
    p : Tensor
        Probabilities in [0, 1].
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        Quantiles.
    """
    return torch.ops.torchscience.gamma_quantile(p, shape, scale)
