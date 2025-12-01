import torch
from torch import Tensor


def hermite_polynomial_he_weight(
    x: Tensor,
) -> Tensor:
    """Compute Probabilists' Hermite weight function.

    The weight function is w(x) = exp(-x^2/2), which appears in the orthogonality
    relation for probabilists' Hermite polynomials:

        integral_{-inf}^{inf} He_m(x) He_n(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate weight.

    Returns
    -------
    Tensor
        Weight values w(x) = exp(-x^2/2).

    Notes
    -----
    The probabilists' Hermite polynomials are orthogonal with respect to
    w(x) = exp(-x^2/2) on the entire real line (-inf, inf).

    The orthogonality relation is:
        integral_{-inf}^{inf} He_m(x) He_n(x) exp(-x^2/2) dx = sqrt(2*pi) * n! * delta_{mn}

    Examples
    --------
    >>> hermite_polynomial_he_weight(torch.tensor([0.0, 1.0, 2.0]))
    tensor([1.0000, 0.6065, 0.1353])
    """
    # No domain warning since domain is (-inf, inf) and weight is valid everywhere
    return torch.exp(-x * x / 2.0)
