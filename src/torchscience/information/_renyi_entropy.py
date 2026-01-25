"""Renyi entropy implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def renyi_entropy(
    p: Tensor,
    alpha: float,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute Renyi entropy of order alpha.

    The Renyi entropy is a one-parameter generalization of Shannon entropy.

    Mathematical Definition
    -----------------------
    .. math::
        H_\alpha(P) = \frac{1}{1-\alpha} \log\left(\sum_i p_i^\alpha\right)

    Special cases:

    - :math:`\alpha \to 1`: Shannon entropy
    - :math:`\alpha = 0`: Hartley entropy (log of support size)
    - :math:`\alpha = 2`: Collision entropy
    - :math:`\alpha \to \infty`: Min-entropy :math:`-\log(\max_i p_i)`

    Parameters
    ----------
    p : Tensor
        Probability distribution (or batch of distributions).
    alpha : float
        Order of Renyi entropy. Must be >= 0 and != 1.
        For alpha close to 1, use shannon_entropy instead.
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors.
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply.
    base : float or None, default=None
        Logarithm base (None for natural log).

    Returns
    -------
    Tensor
        Renyi entropy values.

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> torchscience.information.renyi_entropy(p, alpha=2)
    tensor(1.3863)  # Same as Shannon for uniform

    >>> # Collision entropy (alpha=2)
    >>> p = torch.tensor([0.7, 0.2, 0.1])
    >>> torchscience.information.renyi_entropy(p, alpha=2)
    tensor(0.7765)

    Notes
    -----
    - Renyi entropy is non-increasing in alpha: H_alpha1 >= H_alpha2 if alpha1 < alpha2
    - Supports first-order gradients (second-order TODO)

    See Also
    --------
    shannon_entropy : Special case at alpha=1.
    tsallis_entropy : Related generalized entropy.
    """
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")

    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if abs(alpha - 1.0) < 1e-6:
        raise ValueError(
            "alpha cannot be 1 (use shannon_entropy for this case)"
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
        raise ValueError(f"base must be positive and not 1, got {base}")

    p_dim = p.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

    return torch.ops.torchscience.renyi_entropy(
        p, alpha, dim, input_type, reduction, base
    )
