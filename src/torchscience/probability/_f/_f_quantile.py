"""F-distribution quantile function."""

import torch
from torch import Tensor


def f_quantile(p: Tensor, dfn: Tensor | float, dfd: Tensor | float) -> Tensor:
    r"""Percent point function (quantile function) of the F-distribution.

    Returns :math:`x` such that :math:`P(X \le x) = p` for :math:`X \sim F(d_1, d_2)`.

    Parameters
    ----------
    p : Tensor
        Probability values in (0, 1).
    dfn : Tensor or float
        Numerator degrees of freedom :math:`d_1`. Must be positive.
    dfd : Tensor or float
        Denominator degrees of freedom :math:`d_2`. Must be positive.

    Returns
    -------
    Tensor
        Quantile values.

    Examples
    --------
    >>> p = torch.tensor([0.05, 0.5, 0.95])
    >>> f_quantile(p, dfn=5.0, dfd=10.0)
    tensor([0.2621, 0.9356, 3.3258])

    Notes
    -----
    The implementation uses bisection followed by Newton-Raphson refinement
    for robust convergence.

    See Also
    --------
    f_cumulative_distribution : Inverse of PPF
    """
    dfn_t = (
        dfn
        if isinstance(dfn, Tensor)
        else torch.as_tensor(dfn, dtype=p.dtype, device=p.device)
    )
    dfd_t = (
        dfd
        if isinstance(dfd, Tensor)
        else torch.as_tensor(dfd, dtype=p.dtype, device=p.device)
    )
    return torch.ops.torchscience.f_quantile(p, dfn_t, dfd_t)
