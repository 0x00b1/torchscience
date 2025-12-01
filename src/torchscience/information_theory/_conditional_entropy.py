"""Conditional entropy implementation."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def conditional_entropy(
    joint: Tensor,
    *,
    condition_dim: int = -2,
    target_dim: int = -1,
    input_type: Literal["probability", "log_probability"] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute conditional entropy H(Y|X) from joint probability distribution.

    The conditional entropy measures the remaining uncertainty in Y given
    knowledge of X.

    Mathematical Definition
    -----------------------
    For a joint distribution P(X, Y):

    .. math::
        H(Y|X) = -\sum_{x,y} p(x,y) \log p(y|x)
               = -\sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)}
               = H(X, Y) - H(X)

    where :math:`p(x) = \sum_y p(x,y)` is the marginal distribution.

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution P(X, Y). The dimensions specified by
        ``condition_dim`` and ``target_dim`` contain the joint distribution.
    condition_dim : int, default=-2
        Dimension indexing the conditioning variable X.
    target_dim : int, default=-1
        Dimension indexing the target variable Y.
    input_type : {"probability", "log_probability"}, default="probability"
        How to interpret input tensors:

        - ``"probability"``: Direct probability values (epsilon-clamped)
        - ``"log_probability"``: Log-probabilities (exponentiated before use)
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample conditional entropies
        - ``"mean"``: Mean over batch dimensions
        - ``"sum"``: Sum over batch dimensions
    base : float or None, default=None
        Logarithm base for entropy calculation:

        - ``None``: Natural logarithm (entropy in nats)
        - ``2``: Base-2 logarithm (entropy in bits)
        - ``10``: Base-10 logarithm

    Returns
    -------
    Tensor
        Conditional entropy values H(Y|X). Shape depends on ``reduction``.

    Examples
    --------
    >>> # For independent variables, H(Y|X) = H(Y)
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> p_y = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)  # Independent
    >>> torchscience.information_theory.conditional_entropy(joint)
    tensor(1.3863)  # = H(Y) = log(4)

    >>> # When Y is determined by X, H(Y|X) = 0
    >>> joint = torch.tensor([[0.5, 0.0], [0.0, 0.5]])  # Y = X
    >>> torchscience.information_theory.conditional_entropy(joint)
    tensor(0.0)

    Notes
    -----
    - Conditional entropy is always non-negative: :math:`H(Y|X) \geq 0`
    - Conditioning reduces entropy: :math:`H(Y|X) \leq H(Y)`
    - Chain rule: :math:`H(X, Y) = H(X) + H(Y|X)`
    - For independent variables: :math:`H(Y|X) = H(Y)`
    - Supports first and second-order gradients

    See Also
    --------
    joint_entropy : Joint entropy H(X, Y).
    shannon_entropy : Marginal entropy H(X).
    mutual_information : Mutual information I(X; Y) = H(Y) - H(Y|X).
    """
    if not isinstance(joint, Tensor):
        raise TypeError(f"joint must be a Tensor, got {type(joint).__name__}")

    valid_input_types = ("probability", "log_probability")
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

    ndim = joint.dim()
    if ndim < 2:
        raise ValueError("joint must have at least 2 dimensions")

    # Normalize dims
    if condition_dim < -ndim or condition_dim >= ndim:
        raise IndexError(
            f"condition_dim {condition_dim} out of range for tensor with {ndim} dimensions"
        )
    if target_dim < -ndim or target_dim >= ndim:
        raise IndexError(
            f"target_dim {target_dim} out of range for tensor with {ndim} dimensions"
        )

    norm_cond = condition_dim if condition_dim >= 0 else ndim + condition_dim
    norm_targ = target_dim if target_dim >= 0 else ndim + target_dim

    if norm_cond == norm_targ:
        raise ValueError("condition_dim and target_dim must be different")

    return torch.ops.torchscience.conditional_entropy(
        joint, norm_cond, norm_targ, input_type, reduction, base
    )
