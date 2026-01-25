"""Renyi divergence implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def renyi_divergence(
    p: Tensor,
    q: Tensor,
    alpha: float,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
    base: float | None = None,
    pairwise: bool = False,
) -> Tensor:
    r"""Compute Renyi divergence of order alpha between probability distributions.

    The Renyi divergence is a generalization of the Kullback-Leibler divergence
    that includes a family of divergences parameterized by alpha.

    Mathematical Definition
    -----------------------
    For discrete distributions P and Q:

    .. math::
        D_\alpha(P \| Q) = \frac{1}{\alpha-1} \log\left(\sum_i p_i^\alpha q_i^{1-\alpha}\right)

    Special cases:

    - :math:`\alpha \to 0`: ``-log(P(q > 0))`` (min-divergence)
    - :math:`\alpha \to 1`: KL divergence (limit)
    - :math:`\alpha = 0.5`: Bhattacharyya divergence relation
    - :math:`\alpha = 2`: Chi-squared divergence relation
    - :math:`\alpha \to \infty`: Max-divergence

    Parameters
    ----------
    p : Tensor
        First probability distribution (or batch of distributions).
    q : Tensor
        Second probability distribution (or batch of distributions).
    alpha : float
        Order of the Renyi divergence. Must be >= 0 and != 1.
        For alpha close to 1, use kullback_leibler_divergence instead.
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
        - ``"batchmean"``: Mean over batch dimension
        - ``"sum"``: Sum over all elements
    base : float or None, default=None
        Logarithm base. If None, uses natural logarithm (base e).
        Common choices: 2 (bits), e (nats), 10 (hartleys).
    pairwise : bool, default=False
        If True, compute all-pairs divergence matrix.
        ``p: (m, n)`` and ``q: (k, n)`` produces output ``(m, k)``.

    Returns
    -------
    Tensor
        Renyi divergence values. Shape depends on ``reduction`` and ``pairwise``.

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> q = torch.tensor([0.1, 0.2, 0.3, 0.4])
    >>> torchscience.information.renyi_divergence(p, q, alpha=2.0)
    tensor(0.2157)

    >>> # Alpha=0.5 relates to Bhattacharyya distance
    >>> d_half = torchscience.information.renyi_divergence(p, q, alpha=0.5)

    >>> # Batch of distributions
    >>> p = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> q = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> d = torchscience.information.renyi_divergence(
    ...     p, q, alpha=2.0, reduction="none"
    ... )
    >>> d.shape
    torch.Size([10])

    Notes
    -----
    - Renyi divergence is asymmetric: :math:`D_\\alpha(P \| Q) \\neq D_\\alpha(Q \| P)`
    - Unlike KL divergence, Renyi divergence is finite when Q is zero where P is nonzero (for alpha < 1)
    - Supports first-order gradients

    See Also
    --------
    kullback_leibler_divergence : Special case at alpha=1.
    jensen_shannon_divergence : Symmetric divergence.
    renyi_entropy : Related generalized entropy.
    """
    # Validate input types
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")
    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")

    # Validate alpha
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if abs(alpha - 1.0) < 1e-6:
        raise ValueError(
            "alpha cannot be 1 (use kullback_leibler_divergence for this case)"
        )

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

    # Validate base
    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
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

    return torch.ops.torchscience.renyi_divergence(
        p, q, alpha, dim, input_type, reduction, base, pairwise
    )
