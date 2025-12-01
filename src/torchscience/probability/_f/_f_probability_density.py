"""F-distribution probability density function."""

import torch
from torch import Tensor


def f_probability_density(
    x: Tensor, dfn: Tensor | float, dfd: Tensor | float
) -> Tensor:
    r"""Probability density function of the F-distribution.

    .. math::
        f(x; d_1, d_2) = \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}{(d_1 x + d_2)^{d_1+d_2}}}}{x \, B(d_1/2, d_2/2)}

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the PDF. Must be non-negative.
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        PDF values.

    Examples
    --------
    >>> x = torch.linspace(0.1, 3, 5)
    >>> f_probability_density(x, dfn=5.0, dfd=10.0)
    tensor([0.2156, 0.6694, 0.4579, 0.2013, 0.0719])

    Notes
    -----
    The mode occurs at :math:`x = \frac{d_1 - 2}{d_1} \cdot \frac{d_2}{d_2 + 2}`
    for :math:`d_1 > 2`.

    See Also
    --------
    f_cumulative_distribution : Cumulative distribution function
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
    return torch.ops.torchscience.f_probability_density(x, dfn_t, dfd_t)
