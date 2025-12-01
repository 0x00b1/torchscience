"""Tsallis entropy implementation."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def tsallis_entropy(
    p: Tensor,
    q: float,
    *,
    dim: int = -1,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
) -> Tensor:
    r"""Compute Tsallis entropy of order q.

    The Tsallis entropy is a generalization of Shannon entropy that is
    non-extensive for q != 1.

    Mathematical Definition
    -----------------------
    .. math::
        S_q(P) = \frac{1}{q-1}\left(1 - \sum_i p_i^q\right)

    Special cases:

    - :math:`q \to 1`: Shannon entropy
    - :math:`q = 2`: Related to purity :math:`\text{Tr}(\rho^2)`

    Properties:

    - Non-extensive: :math:`S_q(A+B) = S_q(A) + S_q(B) + (1-q)S_q(A)S_q(B)`
    - :math:`q < 1`: Favors rare events (sub-extensive)
    - :math:`q > 1`: Favors common events (super-extensive)

    Parameters
    ----------
    p : Tensor
        Probability distribution (or batch of distributions).
    q : float
        Order of Tsallis entropy. Must be != 1.
        For q close to 1, use shannon_entropy instead.
    dim : int, default=-1
        Dimension along which the probability distribution is defined.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        How to interpret input tensors.
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply.

    Returns
    -------
    Tensor
        Tsallis entropy values.

    Examples
    --------
    >>> p = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> torchscience.information_theory.tsallis_entropy(p, q=2)
    tensor(0.75)

    >>> # Non-uniform distribution
    >>> p = torch.tensor([0.7, 0.2, 0.1])
    >>> torchscience.information_theory.tsallis_entropy(p, q=2)
    tensor(0.46)

    Notes
    -----
    - The relationship to Renyi entropy: :math:`S_q = (1 - e^{(1-q)H_q}) / (q-1)`
    - Tsallis entropy is used in statistical mechanics for non-extensive systems
    - Supports first-order gradients

    See Also
    --------
    shannon_entropy : Special case at q=1.
    renyi_entropy : Related generalized entropy.
    """
    if not isinstance(p, Tensor):
        raise TypeError(f"p must be a Tensor, got {type(p).__name__}")

    if abs(q - 1.0) < 1e-6:
        raise ValueError("q cannot be 1 (use shannon_entropy for this case)")

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

    return torch.ops.torchscience.tsallis_entropy(
        p, q, dim, input_type, reduction
    )
