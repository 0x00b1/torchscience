"""F-distribution survival function."""

import torch
from torch import Tensor


def f_survival(x: Tensor, dfn: Tensor | float, dfd: Tensor | float) -> Tensor:
    r"""Survival function (1 - CDF) of the F-distribution.

    .. math::
        S(x; d_1, d_2) = 1 - F(x; d_1, d_2) = I_{1-x'}(d_2/2, d_1/2)

    where :math:`1 - x' = \frac{d_2}{d_1 x + d_2}`.

    More numerically stable than ``1 - f_cumulative_distribution(x, dfn, dfd)`` for large x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function.
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        Survival function values :math:`P(X > x)`.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1.0, 2.0])
    >>> f_survival(x, dfn=5.0, dfd=10.0)
    tensor([0.7789, 0.5000, 0.1594])

    See Also
    --------
    f_cumulative_distribution : CDF = 1 - SF
    """
    dfn_t = (
        dfn
        if isinstance(dfn, Tensor)
        else torch.as_tensor(dfn, dtype=x.dtype, device=x.device)
    )
    dfd_t = (
        dfd
        if isinstance(dfd, Tensor)
        else torch.as_tensor(dfd, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.f_survival(x, dfn_t, dfd_t)
