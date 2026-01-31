"""Poisson log probability mass function."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def poisson_log_probability_mass(k: Tensor, rate: Tensor) -> Tensor:
    r"""Log probability mass function of the Poisson distribution.

    .. math::
        \log P(X = k) = k \log(\lambda) - \lambda - \log(\Gamma(k + 1))

    Parameters
    ----------
    k : Tensor
        Number of events.
    rate : Tensor
        Rate parameter (lambda).

    Returns
    -------
    Tensor
        Log probability :math:`\log P(X = k)`.

    Notes
    -----
    - For k < 0, returns -inf.
    - For rate <= 0 with k = 0, returns 0.
    - For rate <= 0 with k > 0, returns -inf.
    - k is floored to the nearest integer internally.
    - Gradients are computed with respect to both k (continuous relaxation)
      and rate.

    Examples
    --------
    >>> k = torch.arange(0, 10, dtype=torch.float32)
    >>> rate = torch.tensor(5.0)
    >>> poisson_log_probability_mass(k, rate)
    tensor([-5.0000, -3.3906, -2.4740, -1.9633, -1.7398, -1.7398, -1.9227, -2.2587, -2.7276, -3.3155])
    """
    return torch.ops.torchscience.poisson_log_probability_mass(k, rate)
