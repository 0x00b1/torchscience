"""Whittaker W function."""

import torch
from torch import Tensor


def whittaker_w(kappa: Tensor, mu: Tensor, z: Tensor) -> Tensor:
    r"""Whittaker W function W_{κ,μ}(z).

    The Whittaker W function is a solution to Whittaker's equation and is
    related to the confluent hypergeometric function U by:

    .. math::
        W_{\kappa,\mu}(z) = e^{-z/2} z^{\mu+1/2} U(\mu - \kappa + 1/2, 2\mu + 1, z)

    Unlike the M function, W decays exponentially as z → +∞.

    Args:
        kappa: First parameter tensor (κ).
        mu: Second parameter tensor (μ).
        z: Argument tensor.

    Returns:
        The Whittaker W function W_{κ,μ}(z).

    Examples:
        >>> import torch
        >>> from torchscience.special_functions import whittaker_w
        >>> kappa = torch.tensor([0.5])
        >>> mu = torch.tensor([1.0])
        >>> z = torch.tensor([2.0])
        >>> whittaker_w(kappa, mu, z)

    References:
        - DLMF 13.14: https://dlmf.nist.gov/13.14
    """
    return torch.ops.torchscience.whittaker_w(kappa, mu, z)
