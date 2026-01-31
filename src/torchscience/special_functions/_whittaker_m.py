"""Whittaker M function."""

import torch
from torch import Tensor


def whittaker_m(kappa: Tensor, mu: Tensor, z: Tensor) -> Tensor:
    r"""Whittaker M function M_{κ,μ}(z).

    The Whittaker M function is a solution to Whittaker's equation and is
    related to the confluent hypergeometric function M by:

    .. math::
        M_{\kappa,\mu}(z) = e^{-z/2} z^{\mu+1/2} M(\mu - \kappa + 1/2, 2\mu + 1, z)

    Args:
        kappa: First parameter tensor (κ).
        mu: Second parameter tensor (μ).
        z: Argument tensor.

    Returns:
        The Whittaker M function M_{κ,μ}(z).

    Examples:
        >>> import torch
        >>> from torchscience.special_functions import whittaker_m
        >>> kappa = torch.tensor([0.5])
        >>> mu = torch.tensor([1.0])
        >>> z = torch.tensor([2.0])
        >>> whittaker_m(kappa, mu, z)

    References:
        - DLMF 13.14: https://dlmf.nist.gov/13.14
    """
    return torch.ops.torchscience.whittaker_m(kappa, mu, z)
