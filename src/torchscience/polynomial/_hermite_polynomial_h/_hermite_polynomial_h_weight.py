import torch
from torch import Tensor


def hermite_polynomial_h_weight(
    x: Tensor,
) -> Tensor:
    """Compute Physicists' Hermite weight function.

    The weight function is w(x) = exp(-x^2), which appears in the orthogonality
    relation for physicists' Hermite polynomials:

        integral_{-inf}^{inf} H_m(x) H_n(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate weight.

    Returns
    -------
    Tensor
        Weight values w(x) = exp(-x^2).

    Notes
    -----
    The physicists' Hermite polynomials are orthogonal with respect to
    w(x) = exp(-x^2) on the entire real line (-inf, inf).

    The orthogonality relation is:
        integral_{-inf}^{inf} H_m(x) H_n(x) exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_{mn}

    Examples
    --------
    >>> hermite_polynomial_h_weight(torch.tensor([0.0, 1.0, 2.0]))
    tensor([1.0000, 0.3679, 0.0183])
    """
    # No domain warning since domain is (-inf, inf) and weight is valid everywhere
    return torch.exp(-x * x)
