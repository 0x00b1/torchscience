"""f-divergence implementation."""

from typing import Callable, Literal

import torch
from torch import Tensor


def f_divergence(
    p: Tensor,
    q: Tensor,
    f: Callable[[Tensor], Tensor],
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "batchmean", "sum"] = "none",
) -> Tensor:
    r"""Compute f-divergence with custom generator function.

    The f-divergence is a family of divergences parameterized by a convex
    function f with f(1) = 0.

    Mathematical Definition
    -----------------------
    .. math::
        D_f(P \| Q) = \sum_i q_i \cdot f\left(\frac{p_i}{q_i}\right)

    Common choices for f:

    - f(t) = t log(t): KL divergence
    - f(t) = -log(t): Reverse KL
    - f(t) = (t-1)^2: Chi-squared divergence
    - f(t) = (sqrt(t) - 1)^2: Squared Hellinger distance
    - f(t) = |t-1|/2: Total variation distance

    Parameters
    ----------
    p : Tensor
        First probability distribution (or batch of distributions).
    q : Tensor
        Second probability distribution (or batch of distributions).
    f : Callable[[Tensor], Tensor]
        Generator function. Must be a convex function with f(1) = 0 for
        the divergence to be well-defined. The function should accept
        and return tensors, and should be compatible with autograd.
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

    Returns
    -------
    Tensor
        f-divergence values. Shape depends on ``reduction``.

    Examples
    --------
    >>> # KL divergence via f-divergence
    >>> def kl_generator(t):
    ...     return t * torch.log(t)
    >>> p = torch.tensor([0.4, 0.6])
    >>> q = torch.tensor([0.5, 0.5])
    >>> torchscience.information.f_divergence(p, q, kl_generator)
    tensor(0.0204)

    >>> # Chi-squared divergence
    >>> def chi_squared_generator(t):
    ...     return (t - 1) ** 2
    >>> torchscience.information.f_divergence(p, q, chi_squared_generator)
    tensor(0.0400)

    >>> # Batch of distributions
    >>> p = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> q = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> def f(t):
    ...     return t * torch.log(t)
    >>> d = torchscience.information.f_divergence(p, q, f)
    >>> d.shape
    torch.Size([10])

    Notes
    -----
    - Autograd support comes from PyTorch's autograd on the user's f function.
    - For optimal performance, use the specialized operators (kullback_leibler_divergence,
      chi_squared_divergence, etc.) instead of f_divergence when available.
    - The function f must satisfy f(1) = 0 for the divergence to equal zero
      when p = q. This is not validated at runtime.

    See Also
    --------
    kullback_leibler_divergence : Specialized KL divergence operator.
    chi_squared_divergence : Specialized chi-squared divergence operator.
    jensen_shannon_divergence : Specialized Jensen-Shannon divergence operator.
    """
    # Validate input types
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")
    if not isinstance(q, Tensor):
        raise TypeError(f"q must be a Tensor, got {type(q).__name__}")
    if not callable(f):
        raise TypeError(f"f must be a callable, got {type(f).__name__}")

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

    # Preprocess inputs based on input_type
    if input_type == "probability":
        p_prob = p
        q_prob = q
    elif input_type == "log_probability":
        p_prob = p.exp()
        q_prob = q.exp()
    else:  # logits
        p_prob = torch.softmax(p, dim=dim)
        q_prob = torch.softmax(q, dim=dim)

    # Dtype promotion
    target_dtype = torch.promote_types(p_prob.dtype, q_prob.dtype)
    if p_prob.dtype != target_dtype:
        p_prob = p_prob.to(target_dtype)
    if q_prob.dtype != target_dtype:
        q_prob = q_prob.to(target_dtype)

    # Normalize dim
    p_dim = p_prob.dim()
    if dim < -p_dim or dim >= p_dim:
        raise IndexError(
            f"dim {dim} out of range for tensor with {p_dim} dimensions"
        )
    dim = dim if dim >= 0 else p_dim + dim

    # Check distribution dimension sizes match
    if p_prob.size(dim) != q_prob.size(dim):
        raise ValueError(
            f"Distribution sizes must match along dim {dim}: "
            f"p has {p_prob.size(dim)}, q has {q_prob.size(dim)}"
        )

    # Compute f-divergence: D_f(P || Q) = sum_i q_i * f(p_i / q_i)
    eps = torch.finfo(p_prob.dtype).eps * 10
    q_safe = q_prob.clamp(min=eps)
    p_safe = p_prob.clamp(min=eps)

    ratio = p_safe / q_safe
    f_values = f(ratio)

    divergence = (q_prob * f_values).sum(dim=dim)

    # Apply reduction
    if reduction == "none":
        return divergence
    elif reduction == "sum":
        return divergence.sum()
    elif reduction == "mean":
        return divergence.mean()
    else:  # batchmean
        batch_size = divergence.numel()
        return (
            divergence.sum() / batch_size
            if batch_size > 0
            else divergence.sum()
        )
