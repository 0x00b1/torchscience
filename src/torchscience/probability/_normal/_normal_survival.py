"""Normal survival function."""

import torch
from torch import Tensor


def normal_survival(
    x: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Survival function (1 - CDF) of the normal distribution.

    .. math::
        S(x) = P(X > x) = 1 - \Phi(x) = \frac{1}{2} \text{erfc}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)

    More numerically stable than ``1 - normal_cumulative_distribution(x)`` for large x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function.
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        Survival function values.

    Examples
    --------
    >>> x = torch.tensor([0.0, 1.0, 2.0])
    >>> normal_survival(x)
    tensor([0.5000, 0.1587, 0.0228])

    See Also
    --------
    normal_cumulative_distribution : CDF = 1 - SF
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
    return torch.ops.torchscience.normal_survival(x, loc_t, scale_t)
