"""Chi-squared cumulative distribution function."""

import torch
from torch import Tensor


def chi2_cumulative_distribution(x: Tensor, df: Tensor | float) -> Tensor:
    r"""Cumulative distribution function of the chi-squared distribution.

    .. math::
        F(x; k) = P(k/2, x/2) = \frac{\gamma(k/2, x/2)}{\Gamma(k/2)}

    where :math:`P(a, x)` is the regularized lower incomplete gamma function.

    Parameters
    ----------
    x : Tensor
        Quantiles. Must be non-negative.
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        CDF values :math:`P(X \le x)` where :math:`X \sim \chi^2(k)`.

    Examples
    --------
    >>> x = torch.tensor([1.0, 5.0, 10.0])
    >>> df = torch.tensor(5.0)
    >>> chi2_cumulative_distribution(x, df)
    tensor([0.0374, 0.5841, 0.9247])

    Notes
    -----
    For :math:`k = 2`, the chi-squared distribution reduces to the exponential
    distribution with rate :math:`\lambda = 1/2`:

    .. math::
        F(x; 2) = 1 - e^{-x/2}

    See Also
    --------
    chi2_probability_density : Probability density function
    chi2_quantile : Inverse CDF (quantile function)
    chi2_survival : Survival function (1 - CDF)
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.chi2_cumulative_distribution(x, df_t)
