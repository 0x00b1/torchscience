"""F-distribution cumulative distribution function."""

import torch
from torch import Tensor


def f_cumulative_distribution(
    x: Tensor, dfn: Tensor | float, dfd: Tensor | float
) -> Tensor:
    r"""Cumulative distribution function of the F-distribution.

    .. math::
        F(x; d_1, d_2) = I_{x'}(d_1/2, d_2/2)

    where :math:`x' = \frac{d_1 x}{d_1 x + d_2}` and :math:`I_x(a,b)` is the
    regularized incomplete beta function.

    Parameters
    ----------
    x : Tensor
        Quantiles. Must be non-negative.
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        CDF values :math:`P(X \le x)` where :math:`X \sim F(d_1, d_2)`.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1.0, 2.0])
    >>> f_cumulative_distribution(x, dfn=5.0, dfd=10.0)
    tensor([0.2211, 0.5000, 0.8406])

    Notes
    -----
    When :math:`d_1 = d_2`, the CDF at :math:`x = 1` equals 0.5 (the median).

    The F-distribution arises as the ratio of two independent chi-squared
    random variables divided by their degrees of freedom:

    .. math::
        F = \frac{X_1 / d_1}{X_2 / d_2}

    where :math:`X_1 \sim \chi^2(d_1)` and :math:`X_2 \sim \chi^2(d_2)`.

    See Also
    --------
    f_probability_density : Probability density function
    f_quantile : Inverse CDF (quantile function)
    f_survival : Survival function (1 - CDF)
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
    return torch.ops.torchscience.f_cumulative_distribution(x, dfn_t, dfd_t)
