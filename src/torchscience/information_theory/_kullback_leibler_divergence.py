"""Kullback-Leibler divergence implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def kullback_leibler_divergence(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
    pairwise: bool = False,
) -> Tensor:
    r"""Compute Kullback-Leibler divergence between probability distributions.

    The KL divergence measures how one probability distribution P diverges
    from a reference distribution Q.

    Mathematical Definition
    -----------------------
    For discrete distributions P and Q:

    .. math::
        D_{KL}(P \| Q) = \sum_{i} p_i \log\left(\frac{p_i}{q_i}\right)

    Parameters
    ----------
    p : Tensor
        First probability distribution (or batch of distributions).
    q : Tensor
        Second probability distribution (or batch of distributions).
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors:

        - ``"probability"``: Direct probability mass functions (epsilon-clamped)
        - ``"log_probability"``: Log-probabilities (exponentiated before use)
        - ``"logits"``: Unnormalized logits (softmax applied)
    reduction : {"none", "mean", "batchmean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample divergences
        - ``"mean"``: Mean over all elements
        - ``"batchmean"``: Mean over batch dimension (mathematically correct KL)
        - ``"sum"``: Sum over all elements
    pairwise : bool, default=False
        If True, compute all-pairs divergence matrix.
        ``p: (m, n)`` and ``q: (k, n)`` produces output ``(m, k)``.

    Returns
    -------
    Tensor
        KL divergence values. Shape depends on ``reduction`` and ``pairwise``.

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    >>> torchscience.information_theory.kullback_leibler_divergence(p, q)
    tensor(0.1335)

    >>> # Batch of distributions
    >>> p = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> q = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> kl = torchscience.information_theory.kullback_leibler_divergence(
    ...     p, q, reduction="none"
    ... )
    >>> kl.shape
    torch.Size([10])

    Notes
    -----
    - KL divergence is asymmetric: :math:`D_{KL}(P \| Q) \\neq D_{KL}(Q \| P)`
    - Values are clamped to avoid log(0): ``eps`` is dtype-dependent
    - Supports first and second-order gradients

    See Also
    --------
    jensen_shannon_divergence : Symmetric divergence measure.
    torch.nn.functional.kl_div : PyTorch's KL divergence (different API).
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

    return torch.ops.torchscience.kullback_leibler_divergence(
        p, q, dim, input_type, reduction, pairwise
    )
