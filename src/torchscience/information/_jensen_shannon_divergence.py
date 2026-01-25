"""Jensen-Shannon divergence implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def jensen_shannon_divergence(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
    pairwise: bool = False,
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute Jensen-Shannon divergence between probability distributions.

    The JS divergence is a symmetric, bounded measure based on KL divergence.

    Mathematical Definition
    -----------------------
    .. math::
        D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)

    where :math:`M = \frac{1}{2}(P + Q)` is the mixture distribution.

    Parameters
    ----------
    p : Tensor
        First probability distribution (or batch of distributions).
    q : Tensor
        Second probability distribution (or batch of distributions).
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors.
    reduction : {"none", "mean", "batchmean", "sum"}, default="none"
        Reduction to apply.
    pairwise : bool, default=False
        If True, compute all-pairs divergence matrix.
    base : float, optional
        Logarithm base for output scaling. ``None`` for natural log (nats),
        ``2`` for bits. JS divergence is bounded by ``log(2)`` in the
        specified base.

    Returns
    -------
    Tensor
        JS divergence values.

    Examples
    --------
    >>> p = torch.tensor([0.5, 0.5])
    >>> q = torch.tensor([0.1, 0.9])
    >>> torchscience.information.jensen_shannon_divergence(p, q)
    tensor(0.1927)

    >>> # In bits (bounded by 1.0)
    >>> torchscience.information.jensen_shannon_divergence(p, q, base=2)
    tensor(0.2780)

    Notes
    -----
    - JS divergence is symmetric: :math:`D_{JS}(P \| Q) = D_{JS}(Q \| P)`
    - Bounded: :math:`0 \\leq D_{JS} \\leq \\log(2)` (in nats)
    - Square root of JS divergence is a proper metric

    See Also
    --------
    kullback_leibler_divergence : Asymmetric divergence measure.
    """
    # Validate input types
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")
    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")

    # Validate input_type
    valid_input_types = ("probability", "log_probability", "logits")
    if input_type not in valid_input_types:
        raise ValueError(
            f"input_type must be one of {valid_input_types}, got '{input_type}'"
        )

    # Validate reduction
    valid_reductions = ("none", "mean", "batchmean", "sum")
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )

    # Normalize dim
    p_dim = p.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

    # Check distribution dimension sizes match
    if p.size(dim) != q.size(dim):
        raise ValueError(
            f"Distribution sizes must match along dim {dim}: "
            f"p has {p.size(dim)}, q has {q.size(dim)}"
        )

    # Validate pairwise mode
    if pairwise:
        if p.dim() < 2 or q.dim() < 2:
            raise ValueError(
                "pairwise=True requires p and q to be at least 2D"
            )

    # Dtype promotion
    target_dtype = torch.promote_types(p.dtype, q.dtype)
    if p.dtype != target_dtype:
        p = p.to(target_dtype)
    if q.dtype != target_dtype:
        q = q.to(target_dtype)

    return torch.ops.torchscience.jensen_shannon_divergence(
        p, q, dim, input_type, reduction, base, pairwise
    )
