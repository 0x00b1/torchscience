"""Poisson probability mass function."""

import torch
from torch import Tensor


def poisson_probability_mass(k: Tensor, rate: Tensor) -> Tensor:
    r"""Probability mass function of the Poisson distribution.

    .. math::
        P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}

    Parameters
    ----------
    k : Tensor
        Number of events.
    rate : Tensor
        Rate parameter.

    Returns
    -------
    Tensor
        Probability :math:`P(X = k)`.

    Notes
    -----
    Gradients are computed with respect to rate only (k is discrete).

    Examples
    --------
    >>> k = torch.arange(0, 10, dtype=torch.float32)
    >>> rate = torch.tensor(5.0)
    >>> poisson_probability_mass(k, rate)
    tensor([0.0067, 0.0337, 0.0842, 0.1404, 0.1755, 0.1755, 0.1462, 0.1044, 0.0653, 0.0363])
    """
    return torch.ops.torchscience.poisson_probability_mass(k, rate)
