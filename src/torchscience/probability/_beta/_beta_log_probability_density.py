"""Beta log probability density function."""

import torch
from torch import Tensor


def beta_log_probability_density(x: Tensor, a: Tensor, b: Tensor) -> Tensor:
    r"""Log probability density function of the beta distribution.

    Computed directly for numerical stability (not as log(pdf)).

    .. math::
        \log f(x; a, b) = (a-1) \log x + (b-1) \log(1-x) - \log B(a, b)

    where :math:`B(a, b)` is the beta function.

    Parameters
    ----------
    x : Tensor
        Values in (0, 1).
    a : Tensor
        First shape parameter. Must be positive.
    b : Tensor
        Second shape parameter. Must be positive.

    Returns
    -------
    Tensor
        Log PDF values.

    Examples
    --------
    >>> x = torch.tensor([0.2, 0.5, 0.8])
    >>> a = torch.tensor(2.0)
    >>> b = torch.tensor(5.0)
    >>> beta_log_probability_density(x, a, b)
    tensor([ 0.5878,  0.2877, -1.5606])

    See Also
    --------
    beta_probability_density : Exp of log PDF
    """
    return torch.ops.torchscience.beta_log_probability_density(x, a, b)
