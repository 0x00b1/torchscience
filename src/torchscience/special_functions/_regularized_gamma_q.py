"""Regularized upper incomplete gamma function Q(a, x)."""

import torch
from torch import Tensor

__all__ = ["regularized_gamma_q"]


def regularized_gamma_q(a: Tensor, x: Tensor) -> Tensor:
    r"""Regularized upper incomplete gamma function.

    Computes the regularized upper incomplete gamma function:

    .. math::

        Q(a, x) = \frac{\Gamma(a, x)}{\Gamma(a)} = 1 - P(a, x)

    where :math:`\Gamma(a, x)` is the upper incomplete gamma function and
    :math:`P(a, x)` is the regularized lower incomplete gamma function.

    The function is implemented using the relationship :math:`Q(a, x) = 1 - P(a, x)`
    where P uses series expansion for :math:`x < a + 1` and continued fraction
    (Lentz's method) for :math:`x \geq a + 1`.

    Parameters
    ----------
    a : Tensor
        Shape parameter. Must be positive.
    x : Tensor
        Integration limit. Must be non-negative.

    Returns
    -------
    Tensor
        The regularized upper incomplete gamma function Q(a, x).

    Notes
    -----
    - Q(a, x) is the survival function (1 - CDF) of the gamma distribution.
    - Q(1, x) = exp(-x) is the exponential survival function.
    - Q(a, 0) = 1 for all a > 0.
    - As x -> infinity, Q(a, x) -> 0.

    Examples
    --------
    >>> import torch
    >>> from torchscience.special_functions import regularized_gamma_q
    >>> a = torch.tensor([1.0, 2.0, 3.0])
    >>> x = torch.tensor([0.5, 1.0, 2.0])
    >>> regularized_gamma_q(a, x)
    tensor([0.6065, 0.7358, 0.6767])

    >>> # Exponential survival function: Q(1, x) = exp(-x)
    >>> x = torch.tensor([1.0, 2.0])
    >>> regularized_gamma_q(torch.ones_like(x), x)
    tensor([0.3679, 0.1353])

    See Also
    --------
    regularized_gamma_p : Regularized lower incomplete gamma function P(a, x).
    """
    return torch.ops.torchscience.regularized_gamma_q(a, x)
