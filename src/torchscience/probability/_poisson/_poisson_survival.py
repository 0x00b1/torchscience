"""Poisson survival function."""

import torch
from torch import Tensor


def poisson_survival(k: Tensor, rate: Tensor) -> Tensor:
    r"""Survival function of the Poisson distribution.

    .. math::
        S(k; \lambda) = P(X > k) = P(k+1, \lambda)

    where P is the lower regularized incomplete gamma function.

    Parameters
    ----------
    k : Tensor
        Number of events. Non-negative integers (floored if float).
    rate : Tensor
        Rate parameter :math:`\lambda` (mean). Must be positive.

    Returns
    -------
    Tensor
        Survival probability :math:`P(X > k)`.

    Notes
    -----
    The survival function is the complement of the CDF:
    :math:`S(k) = 1 - F(k) = P(X > k)`.

    Gradients are computed with respect to rate only (k is discrete).

    For k < 0, returns 1.0 (all probability mass exceeds k).
    For rate <= 0, returns 0.0.

    Examples
    --------
    >>> k = torch.tensor([0.0, 2.0, 5.0, 10.0])
    >>> rate = torch.tensor(5.0)
    >>> poisson_survival(k, rate)
    tensor([0.9933, 0.8753, 0.3840, 0.0137])
    """
    return torch.ops.torchscience.poisson_survival(k, rate)
