"""Normal quantile function."""

import torch
from torch import Tensor


def normal_quantile(
    p: Tensor,
    loc: Tensor | float = 0.0,
    scale: Tensor | float = 1.0,
) -> Tensor:
    r"""Percent point function (quantile function, inverse CDF) of the normal distribution.

    Returns x such that :math:`P(X \le x) = p` for :math:`X \sim \mathcal{N}(\mu, \sigma^2)`.

    .. math::
        \Phi^{-1}(p; \mu, \sigma) = \mu + \sigma \cdot \Phi^{-1}(p)

    Parameters
    ----------
    p : Tensor
        Probability values in (0, 1).
    loc : Tensor or float, default=0.0
        Mean parameter.
    scale : Tensor or float, default=1.0
        Standard deviation parameter. Must be positive.

    Returns
    -------
    Tensor
        Quantile values.

    Examples
    --------
    >>> p = torch.tensor([0.025, 0.5, 0.975])
    >>> normal_quantile(p)
    tensor([-1.9600,  0.0000,  1.9600])

    See Also
    --------
    normal_cumulative_distribution : Inverse of PPF
    """
    loc_t = (
        loc
        if isinstance(loc, Tensor)
        else torch.as_tensor(loc, dtype=p.dtype, device=p.device)
    )
    scale_t = (
        scale
        if isinstance(scale, Tensor)
        else torch.as_tensor(scale, dtype=p.dtype, device=p.device)
    )
    return torch.ops.torchscience.normal_quantile(p, loc_t, scale_t)
