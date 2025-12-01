"""Chi-squared survival function."""

import torch
from torch import Tensor


def chi2_survival(x: Tensor, df: Tensor | float) -> Tensor:
    r"""Survival function (1 - CDF) of the chi-squared distribution.

    .. math::
        S(x; k) = 1 - F(x; k) = Q(k/2, x/2)

    where :math:`Q(a, x)` is the regularized upper incomplete gamma function.

    More numerically stable than ``1 - chi2_cumulative_distribution(x, df)`` for large x.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the survival function.
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        Survival function values :math:`P(X > x)`.

    Examples
    --------
    >>> x = torch.tensor([1.0, 5.0, 10.0])
    >>> chi2_survival(x, df=5.0)
    tensor([0.9626, 0.4159, 0.0753])

    See Also
    --------
    chi2_cumulative_distribution : CDF = 1 - SF
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.chi2_survival(x, df_t)
