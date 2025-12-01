"""Beta quantile function."""

import torch
from torch import Tensor


def beta_quantile(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Quantile function (inverse CDF) of the beta distribution.

    Parameters
    ----------
    p : Tensor
        Probabilities in [0, 1].
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        Quantiles.
    """
    return torch.ops.torchscience.beta_quantile(p, a, b)
