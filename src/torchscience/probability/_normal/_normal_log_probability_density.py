"""Normal log probability density function."""

import torch
from torch import Tensor


def normal_log_probability_density(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Log of the probability density function of the normal distribution.

    .. math::
        \log \phi(x) = -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(x-\mu)^2}{2\sigma^2}

    More numerically stable than ``log(normal_probability_density(x))`` for extreme x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the log PDF.
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        Log PDF values.

    Examples
    --------
    >>> x = torch.tensor([0.0])
    >>> normal_log_probability_density(x)  # log(0.3989) = -0.9189
    tensor([-0.9189])

    See Also
    --------
    normal_probability_density : Exp of log PDF
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
    return torch.ops.torchscience.normal_log_probability_density(
        x, loc_t, scale_t
    )
