"""Chi-squared divergence between probability distributions."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

__all__ = ["chi_squared_divergence"]


def chi_squared_divergence(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    reduction: Literal["none", "mean", "sum"] = "none",
) -> Tensor:
    r"""Compute chi-squared divergence between two probability distributions.

    The chi-squared divergence (Pearson chi-squared divergence) is defined as:

    .. math::
        \chi^2(P \| Q) = \sum_i \frac{(p_i - q_i)^2}{q_i}

    This is an f-divergence that measures the discrepancy between two probability
    distributions. Unlike KL divergence, chi-squared divergence is symmetric in a
    specific sense: :math:`\chi^2(P \| Q) = \chi^2(Q \| P)` when P = Q, but
    generally asymmetric otherwise.

    Parameters
    ----------
    p : Tensor
        First probability distribution. Must be a valid probability distribution
        (non-negative, sums to 1 along `dim`).
    q : Tensor
        Second probability distribution (reference). Must be a valid probability
        distribution (non-negative, sums to 1 along `dim`). Should have the same
        shape as `p`.
    dim : int, optional
        Dimension along which to compute the divergence. Default: -1.
    reduction : str, optional
        Reduction to apply to the output. One of 'none', 'mean', 'sum'.
        Default: 'none'.

    Returns
    -------
    Tensor
        Chi-squared divergence value(s). If reduction is 'none', returns a tensor
        with one fewer dimension than the input (the `dim` dimension is reduced).

    Notes
    -----
    - The chi-squared divergence is always non-negative and equals zero if and
      only if P = Q.
    - The function adds a small epsilon to q values near zero for numerical
      stability.
    - Supports first and second-order gradients.

    Relationship to other divergences:
    - Related to KL divergence by: :math:`D_{KL}(P \| Q) \leq \log(1 + \chi^2(P \| Q))`
    - Related to Hellinger distance by: :math:`H^2(P, Q) \leq \chi^2(P \| Q) / 2`
    - Related to Total Variation: :math:`TV(P, Q) \leq \sqrt{\chi^2(P \| Q) / 2}`

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> q = torch.tensor([0.5, 0.25, 0.125, 0.125])
    >>> chi_squared_divergence(p, q)
    tensor(0.75)

    >>> # Batched computation
    >>> p = torch.softmax(torch.randn(5, 10), dim=-1)
    >>> q = torch.softmax(torch.randn(5, 10), dim=-1)
    >>> result = chi_squared_divergence(p, q)
    >>> result.shape
    torch.Size([5])

    See Also
    --------
    kullback_leibler_divergence : KL divergence between distributions.
    jensen_shannon_divergence : Symmetric divergence measure.
    """
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")
    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            f"reduction must be one of 'none', 'mean', 'sum', got '{reduction}'"
        )

    return torch.ops.torchscience.chi_squared_divergence(p, q, dim, reduction)
