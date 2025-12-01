"""Chi-squared probability density function."""

import torch
from torch import Tensor


def chi2_probability_density(x: Tensor, df: Tensor | float) -> Tensor:
    r"""Probability density function of the chi-squared distribution.

    .. math::
        f(x; k) = \frac{1}{2^{k/2} \Gamma(k/2)} x^{k/2-1} e^{-x/2}

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the PDF. Must be non-negative.
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        PDF values.

    Examples
    --------
    >>> x = torch.linspace(0, 10, 5)
    >>> chi2_probability_density(x, df=3.0)
    tensor([0.0000, 0.2420, 0.1353, 0.0519, 0.0167])

    Notes
    -----
    The PDF has a mode at :math:`x = k - 2` for :math:`k \ge 2`, and at
    :math:`x = 0` for :math:`k < 2`.

    See Also
    --------
    chi2_cumulative_distribution : Cumulative distribution function
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.chi2_probability_density(x, df_t)
