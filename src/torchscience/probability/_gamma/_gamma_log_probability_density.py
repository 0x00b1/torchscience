"""Gamma log probability density function."""

import torch
from torch import Tensor


def gamma_log_probability_density(
    x: Tensor, shape: Tensor, scale: Tensor
) -> Tensor:
    r"""Log probability density function of the gamma distribution.

    Computed directly for numerical stability (not as log(pdf)).

    .. math::
        \log f(x; k, \theta) = (k-1) \log x - \frac{x}{\theta} - \log\Gamma(k) - k \log\theta

    Parameters
    ----------
    x : Tensor
        Values. Must be positive.
    shape : Tensor
        Shape parameter k (or alpha). Must be positive.
    scale : Tensor
        Scale parameter theta. Must be positive.

    Returns
    -------
    Tensor
        Log PDF values.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> shape = torch.tensor(2.0)
    >>> scale = torch.tensor(1.0)
    >>> gamma_log_probability_density(x, shape, scale)
    tensor([-1.0000, -0.6137, -0.9014])

    See Also
    --------
    gamma_probability_density : Exp of log PDF
    """
    return torch.ops.torchscience.gamma_log_probability_density(
        x, shape, scale
    )
