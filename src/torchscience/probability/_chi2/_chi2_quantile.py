"""Chi-squared quantile function."""

import torch
from torch import Tensor


def chi2_quantile(p: Tensor, df: Tensor | float) -> Tensor:
    r"""Percent point function (quantile function) of the chi-squared distribution.

    Returns :math:`x` such that :math:`P(X \le x) = p` for :math:`X \sim \chi^2(k)`.

    Parameters
    ----------
    p : Tensor
        Probability values in (0, 1).
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        Quantile values.

    Examples
    --------
    >>> p = torch.tensor([0.05, 0.5, 0.95])
    >>> chi2_quantile(p, df=5.0)
    tensor([ 1.1455,  4.3515, 11.0705])

    Notes
    -----
    The implementation uses Wilson-Hilferty approximation for the initial guess
    followed by Newton-Raphson refinement.

    See Also
    --------
    chi2_cumulative_distribution : Inverse of PPF
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=p.dtype, device=p.device)
    )
    return torch.ops.torchscience.chi2_quantile(p, df_t)
