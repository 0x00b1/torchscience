"""Bhattacharyya distance implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401


def bhattacharyya_distance(
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
    r"""Compute Bhattacharyya distance between probability distributions.

    The Bhattacharyya distance measures the similarity of two probability
    distributions via the Bhattacharyya coefficient.

    Mathematical Definition
    -----------------------
    .. math::
        D_B(P, Q) = -\ln(BC(P, Q))

    where the Bhattacharyya coefficient is:

    .. math::
        BC(P, Q) = \sum_i \sqrt{p_i \cdot q_i}

    Properties:
    - Symmetric: :math:`D_B(P, Q) = D_B(Q, P)`
    - Non-negative: :math:`D_B(P, Q) \geq 0`
    - Zero for identical: :math:`D_B(P, P) = 0`

    Related to Hellinger distance:
    - :math:`H^2(P, Q) = 1 - \exp(-D_B(P, Q))`
    - :math:`H^2(P, Q) = 1 - BC(P, Q)`

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
        Bhattacharyya distance values.

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    >>> torchscience.distance.bhattacharyya_distance(p, q)
    tensor(0.0253)

    >>> # Identical distributions have zero distance
    >>> torchscience.distance.bhattacharyya_distance(p, p)
    tensor(0.0)

    Notes
    -----
    - The Bhattacharyya distance is unbounded (can be infinity for non-overlapping
      distributions), unlike the Hellinger distance which is bounded to [0, 1].
    - Supports first-order gradients.

    See Also
    --------
    hellinger_distance : Bounded distance derived from Bhattacharyya coefficient.
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

    p_dim = p.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

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

    target_dtype = torch.promote_types(p.dtype, q.dtype)
    if p.dtype != target_dtype:
        p = p.to(target_dtype)
    if q.dtype != target_dtype:
        q = q.to(target_dtype)

    return torch.ops.torchscience.bhattacharyya_distance(
        p, q, dim, input_type, reduction, pairwise
    )
