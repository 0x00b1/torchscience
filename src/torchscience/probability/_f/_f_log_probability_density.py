"""F-distribution log probability density function."""

import torch
from torch import Tensor


def f_log_probability_density(
    x: Tensor, dfn: Tensor | float, dfd: Tensor | float
) -> Tensor:
    r"""Log probability density function of the F-distribution.

    Computed directly for numerical stability (not as log(pdf)).

    .. math::
        \log f(x; d_1, d_2) = \frac{d_1}{2} \log\frac{d_1}{d_2}
            + \left(\frac{d_1}{2} - 1\right) \log x
            - \frac{d_1 + d_2}{2} \log\left(1 + \frac{d_1 x}{d_2}\right)
            - \log B\left(\frac{d_1}{2}, \frac{d_2}{2}\right)

    where :math:`B(a, b)` is the beta function.

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate the log PDF. Must be positive.
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        Log PDF values.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1.0, 2.0])
    >>> f_log_probability_density(x, dfn=5.0, dfd=10.0)
    tensor([-0.2882, -0.4858, -1.0447])

    See Also
    --------
    f_probability_density : Exp of log PDF
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
    return torch.ops.torchscience.f_log_probability_density(x, dfn_t, dfd_t)
