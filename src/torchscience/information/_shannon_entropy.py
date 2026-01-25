"""Shannon entropy implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def shannon_entropy(
    p: Tensor,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute Shannon entropy of probability distributions.

    The Shannon entropy measures the average information content or
    uncertainty in a probability distribution.

    Mathematical Definition
    -----------------------
    For a discrete probability distribution P:

    .. math::
        H(P) = -\sum_{i} p_i \log(p_i)

    By convention, :math:`0 \log(0) = 0`.

    Parameters
    ----------
    p : Tensor
        Probability distribution (or batch of distributions).
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors:

        - ``"probability"``: Direct probability mass functions (epsilon-clamped)
        - ``"log_probability"``: Log-probabilities (exponentiated before use)
        - ``"logits"``: Unnormalized logits (softmax applied)
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample entropies
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
        Shannon entropy values. Shape depends on ``reduction``.

    Examples
    --------
    >>> # Uniform distribution has maximum entropy
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> torchscience.information.shannon_entropy(p)
    tensor(1.3863)  # log(4) nats

    >>> # In bits
    >>> torchscience.information.shannon_entropy(p, base=2)
    tensor(2.0)  # log2(4) = 2 bits

    >>> # Delta distribution has zero entropy
    >>> p = torch.tensor([1.0, 0.0, 0.0, 0.0])
    >>> torchscience.information.shannon_entropy(p)
    tensor(0.0)

    >>> # Batch of distributions
    >>> p = torch.softmax(torch.randn(10, 5), dim=-1)
    >>> H = torchscience.information.shannon_entropy(p)
    >>> H.shape
    torch.Size([10])

    Notes
    -----
    - Entropy is always non-negative: :math:`H(P) \geq 0`
    - Maximum entropy for n outcomes: :math:`H_{\max} = \log(n)`
    - Values are clamped to avoid log(0): ``eps`` is dtype-dependent
    - Supports first and second-order gradients

    See Also
    --------
    cross_entropy : Cross-entropy between two distributions.
    kullback_leibler_divergence : KL divergence.
    """
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")

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

    return torch.ops.torchscience.shannon_entropy(
        p, dim, input_type, reduction, base
    )
