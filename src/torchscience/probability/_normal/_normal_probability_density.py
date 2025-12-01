"""Normal probability density function."""

import torch
from torch import Tensor


def normal_probability_density(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Probability density function of the normal distribution.

    .. math::
        \phi(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the PDF.
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        PDF values.

    Examples
    --------
    >>> x = torch.tensor([0.0])
    >>> normal_probability_density(x)  # Peak of standard normal
    tensor([0.3989])

    See Also
    --------
    normal_logpdf : Log PDF (more numerically stable)
    normal_cumulative_distribution : Cumulative distribution function
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
    return torch.ops.torchscience.normal_probability_density(x, loc_t, scale_t)
