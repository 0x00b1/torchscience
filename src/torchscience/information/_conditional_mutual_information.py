"""Conditional mutual information operator."""

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def conditional_mutual_information(
    joint: Tensor,
    *,
    dims_x: Tuple[int, ...] = (0,),
    dims_y: Tuple[int, ...] = (1,),
    dims_z: Tuple[int, ...] = (2,),
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute conditional mutual information I(X;Y|Z) from a joint distribution.

    The conditional mutual information measures the amount of information
    that X and Y share about each other, given knowledge of Z:

    .. math::

        I(X;Y|Z) = \sum_{x,y,z} p(x,y,z) \log \frac{p(x,y|z)}{p(x|z) p(y|z)}
                 = H(X|Z) + H(Y|Z) - H(X,Y|Z)

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution tensor p(x,y,z). The tensor should
        have at least 3 dimensions corresponding to the random variables.
    dims_x : Tuple[int, ...], default=(0,)
        Dimensions corresponding to variable X.
    dims_y : Tuple[int, ...], default=(1,)
        Dimensions corresponding to variable Y.
    dims_z : Tuple[int, ...], default=(2,)
        Dimensions corresponding to conditioning variable Z.
    input_type : {"probability", "log_probability", "logits"}, default="probability"
        Type of input values:

        - ``"probability"``: Values are probabilities in [0, 1]
        - ``"log_probability"``: Values are log probabilities
        - ``"logits"``: Values are unnormalized log probabilities
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample values
        - ``"mean"``: Mean over batch dimensions
        - ``"sum"``: Sum over batch dimensions
    base : float or None, default=None
        Logarithm base. None for natural log (nats), 2 for bits, 10 for dits.

    Returns
    -------
    Tensor
        Conditional mutual information I(X;Y|Z). Shape depends on reduction.

    Examples
    --------
    >>> # Independent X and Y given Z
    >>> joint = torch.zeros(2, 2, 2)
    >>> joint[0, 0, 0] = joint[0, 1, 0] = joint[1, 0, 0] = joint[1, 1, 0] = 0.125
    >>> joint[0, 0, 1] = joint[0, 1, 1] = joint[1, 0, 1] = joint[1, 1, 1] = 0.125
    >>> conditional_mutual_information(joint)  # Should be ~0

    Notes
    -----
    - I(X;Y|Z) >= 0 always (non-negative)
    - I(X;Y|Z) = 0 iff X and Y are conditionally independent given Z
    - I(X;Y|Z) can be greater or less than I(X;Y)
    - Supports first-order gradients
    - Currently only single-dimension variables are supported (each of dims_x,
      dims_y, dims_z must contain exactly one dimension index)

    See Also
    --------
    mutual_information : Unconditional mutual information I(X;Y).
    conditional_entropy : Conditional entropy H(Y|X).
    """
    if not isinstance(joint, Tensor):
        raise TypeError(f"joint must be a Tensor, got {type(joint)}")

    if joint.dim() < 3:
        raise ValueError(
            f"joint must have at least 3 dimensions, got {joint.dim()}"
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

    if base is not None and base <= 0:
        raise ValueError(f"base must be positive, got {base}")

    # Validate dimension tuples don't overlap
    all_dims = set(dims_x) | set(dims_y) | set(dims_z)
    if len(all_dims) != len(dims_x) + len(dims_y) + len(dims_z):
        raise ValueError("dims_x, dims_y, and dims_z must not overlap")

    return torch.ops.torchscience.conditional_mutual_information(
        joint,
        list(dims_x),
        list(dims_y),
        list(dims_z),
        input_type,
        reduction,
        base,
    )
