"""Cross-entropy implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def cross_entropy(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute cross-entropy between two probability distributions.

    The cross-entropy measures the average number of bits needed to identify
    an event from a set of possibilities, when using a coding scheme optimized
    for a different (predicted) distribution.

    Mathematical Definition
    -----------------------
    For discrete probability distributions P (true) and Q (predicted):

    .. math::
        H(P, Q) = -\sum_{i} p_i \log(q_i)

    By convention, :math:`0 \log(q) = 0` when :math:`p = 0`.

    Parameters
    ----------
    p : Tensor
        True/target probability distribution (or batch of distributions).
    q : Tensor
        Predicted probability distribution (or batch of distributions).
        Must have the same shape as ``p``.
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors:

        - ``"probability"``: Direct probability mass functions (epsilon-clamped)
        - ``"log_probability"``: Log-probabilities (exponentiated before use)
        - ``"logits"``: Unnormalized logits (softmax applied)
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample cross-entropies
        - ``"mean"``: Mean over all elements
        - ``"sum"``: Sum over all elements
    base : float or None, default=None
        Logarithm base for entropy calculation:

        - ``None``: Natural logarithm (entropy in nats)
        - ``2``: Base-2 logarithm (entropy in bits)
        - ``10``: Base-10 logarithm (entropy in dits/hartleys)

    Returns
    -------
    Tensor
        Cross-entropy values. Shape depends on ``reduction``.

    Examples
    --------
    >>> # Cross-entropy between uniform and a peaked distribution
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> q = torch.tensor([0.7, 0.1, 0.1, 0.1])
    >>> torchscience.information.cross_entropy(p, q)
    tensor(1.6611)

    >>> # Self cross-entropy equals Shannon entropy
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> H_p = torchscience.information.shannon_entropy(p)
    >>> H_pq = torchscience.information.cross_entropy(p, p)
    >>> torch.isclose(H_p, H_pq)
    tensor(True)

    >>> # Batch of distributions
    >>> p = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> q = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> H = torchscience.information.cross_entropy(p, q)
    >>> H.shape
    torch.Size([10])

    Notes
    -----
    - Cross-entropy decomposes as: :math:`H(P, Q) = H(P) + D_{KL}(P \| Q)`
    - Cross-entropy is minimized when :math:`Q = P`, yielding :math:`H(P, P) = H(P)`
    - Unlike KL divergence, cross-entropy is not symmetric
    - Supports first and second-order gradients

    See Also
    --------
    shannon_entropy : Shannon entropy of a single distribution.
    kullback_leibler_divergence : KL divergence between distributions.
    """
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")

    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")

    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p.shape} and {q.shape}"
        )

    valid_input_types = ("probability", "log_probability", "logits")
    if input_type not in valid_input_types:
        raise ValueError(
            f"input_type must be one of {valid_input_types}, got '{input_type}'"
        )

    valid_reductions = ("none", "mean", "sum")
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
        )

    p_dim = p.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

    return torch.ops.torchscience.cross_entropy(
        p, q, dim, input_type, reduction, base
    )
