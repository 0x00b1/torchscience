"""Joint entropy implementation."""

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def joint_entropy(
    joint: Tensor,
    *,
    dims: Tuple[int, ...] = (-2, -1),
    input_type: Literal["probability", "log_probability"] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute joint entropy from joint probability distribution.

    The joint entropy measures the total uncertainty in a joint
    probability distribution over multiple random variables.

    Mathematical Definition
    -----------------------
    For a joint distribution P(X, Y):

    .. math::
        H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)

    By convention, :math:`0 \log(0) = 0`.

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution. The dimensions specified by
        ``dims`` contain the joint distribution, other dimensions are
        batch dimensions.
    dims : tuple of int, default=(-2, -1)
        Dimensions that define the joint distribution. For a 2D joint
        distribution P(X, Y), use the default (-2, -1).
    input_type : {"probability", "log_probability"}, default="probability"
        How to interpret input tensors:

        - ``"probability"``: Direct probability values (epsilon-clamped)
        - ``"log_probability"``: Log-probabilities (exponentiated before use)
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample joint entropies
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
        Joint entropy values. Shape depends on ``reduction`` and ``dims``.

    Examples
    --------
    >>> # Independent uniform distributions
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> p_y = torch.tensor([0.5, 0.5])
    >>> joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)  # Outer product
    >>> torchscience.information.joint_entropy(joint)
    tensor(1.3863)  # H(X) + H(Y) = 2 * log(2)

    >>> # Batched joint distributions
    >>> joint = torch.softmax(torch.randn(10, 4, 5).flatten(-2), dim=-1)
    >>> joint = joint.view(10, 4, 5)
    >>> H = torchscience.information.joint_entropy(joint)
    >>> H.shape
    torch.Size([10])

    Notes
    -----
    - Joint entropy is always non-negative: :math:`H(X, Y) \geq 0`
    - Subadditivity: :math:`H(X, Y) \leq H(X) + H(Y)` with equality
      when X and Y are independent
    - For independent variables: :math:`H(X, Y) = H(X) + H(Y)`
    - Supports first and second-order gradients

    See Also
    --------
    shannon_entropy : Entropy of a single distribution.
    conditional_entropy : Conditional entropy H(Y|X).
    mutual_information : Mutual information I(X; Y).
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
    if len(dims) < 1:
        raise ValueError("dims must have at least 1 element")

    # Normalize dims
    norm_dims = []
    for d in dims:
        if d < -ndim or d >= ndim:
            raise IndexError(
                f"dim {d} out of range for tensor with {ndim} dimensions"
            )
        norm_d = d if d >= 0 else ndim + d
        norm_dims.append(norm_d)

    return torch.ops.torchscience.joint_entropy(
        joint, norm_dims, input_type, reduction, base
    )
