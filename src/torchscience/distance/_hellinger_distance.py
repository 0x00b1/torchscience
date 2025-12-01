"""Hellinger distance implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def hellinger_distance(
    p: Tensor,
    q: Tensor,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    pairwise: bool = False,
) -> Tensor:
    r"""Compute Hellinger distance between probability distributions.

    The Hellinger distance is a symmetric, bounded distance metric
    between probability distributions.

    Mathematical Definition
    -----------------------
    .. math::
        H(P, Q) = \frac{1}{\sqrt{2}} \sqrt{\sum_i (\sqrt{p_i} - \sqrt{q_i})^2}

    Equivalently:
    .. math::
        H(P, Q) = \sqrt{1 - \sum_i \sqrt{p_i q_i}}

    Properties:
    - Symmetric: :math:`H(P, Q) = H(Q, P)`
    - Bounded: :math:`0 \leq H(P, Q) \leq 1`
    - Metric: satisfies the triangle inequality

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
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply.
    pairwise : bool, default=False
        If True, compute all-pairs distance matrix.

    Returns
    -------
    Tensor
        Hellinger distance values.

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    >>> torchscience.distance.hellinger_distance(p, q)
    tensor(0.1548)

    >>> # Identical distributions have zero distance
    >>> torchscience.distance.hellinger_distance(p, p)
    tensor(0.0)

    Notes
    -----
    - Related to Bhattacharyya coefficient: H^2 = 1 - BC
    - Related to Bhattacharyya distance: H^2 = 1 - exp(-D_B)
    - Supports first-order gradients

    See Also
    --------
    bhattacharyya_distance : Related distance measure.
    total_variation_distance : Another bounded distance metric.
    """
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")
    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")

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

    # Normalize dim
    p_dim = p.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

    # Check distribution sizes match
    if p.size(dim) != q.size(dim):
        raise ValueError(
            f"Distribution sizes must match along dim {dim}: "
            f"p has {p.size(dim)}, q has {q.size(dim)}"
        )

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

    return torch.ops.torchscience.hellinger_distance(
        p, q, dim, input_type, reduction, pairwise
    )
