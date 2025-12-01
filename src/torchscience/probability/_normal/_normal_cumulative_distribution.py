"""Normal cumulative distribution function."""

import torch
from torch import Tensor


def normal_cumulative_distribution(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Cumulative distribution function of the normal distribution.

    .. math::
        \Phi(x; \mu, \sigma) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)\right]

    Parameters
    ----------
    x : Tensor
        Quantiles at which to evaluate the CDF.
    loc : Tensor or float, default=0.0
        Mean (location parameter) :math:`\mu`.
    scale : Tensor or float, default=1.0
        Standard deviation (scale parameter) :math:`\sigma`. Must be positive.

    Returns
    -------
    Tensor
        CDF values :math:`P(X \le x)` where :math:`X \sim \mathcal{N}(\mu, \sigma^2)`.

    Examples
    --------
    >>> x = torch.tensor([0.0, 1.0, 2.0])
    >>> normal_cumulative_distribution(x)
    tensor([0.5000, 0.8413, 0.9772])

    >>> normal_cumulative_distribution(x, loc=1.0, scale=0.5)
    tensor([0.0228, 0.5000, 0.9772])

    Notes
    -----
    The gradient with respect to x equals the PDF:

    .. math::
        \frac{\partial \Phi}{\partial x} = \phi(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}

    See Also
    --------
    normal_probability_density : Probability density function
    normal_quantile : Inverse CDF (quantile function)
    normal_survival : Survival function (1 - CDF)
    """
    loc_t = (
        loc
        if isinstance(loc, Tensor)
        else torch.as_tensor(loc, dtype=x.dtype, device=x.device)
    )
    scale_t = (
        scale
        if isinstance(scale, Tensor)
        else torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.normal_cumulative_distribution(
        x, loc_t, scale_t
    )
