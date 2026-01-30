"""Chi-squared log probability density function."""

import torch
from torch import Tensor


def chi2_log_probability_density(x: Tensor, df: Tensor | float) -> Tensor:
    r"""Log probability density function of the chi-squared distribution.

    Computed directly for numerical stability (not as log(pdf)).

    .. math::
        \log f(x; k) = \left(\frac{k}{2} - 1\right) \log x - \frac{x}{2}
        - \frac{k}{2} \log 2 - \log\Gamma\left(\frac{k}{2}\right)

    Parameters
    ----------
    x : Tensor
        Values. Must be positive.
    df : Tensor or float
        Degrees of freedom :math:`k`. Must be positive.

    Returns
    -------
    Tensor
        Log PDF values.

    Examples
    --------
    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> df = torch.tensor(5.0)
    >>> chi2_log_probability_density(x, df)
    tensor([-2.7569, -1.8411, -1.6028])

    See Also
    --------
    chi2_probability_density : Exp of log PDF
    """
    df_t = (
        df
        if isinstance(df, Tensor)
        else torch.as_tensor(df, dtype=x.dtype, device=x.device)
    )
    return torch.ops.torchscience.chi2_log_probability_density(x, df_t)
