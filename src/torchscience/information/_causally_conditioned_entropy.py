"""Causally conditioned entropy operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def causally_conditioned_entropy(
    joint: Tensor,
    *,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute causally conditioned entropy H(Y||X) from a joint distribution.

    The causally conditioned entropy measures the entropy of Y given causal
    access to X - knowing X up to and including time t when predicting Y_t,
    plus the past of Y:

    .. math::

        H(Y \| X) = H(Y_t | Y_{t-1}, X_t)

    This can be computed as:

    .. math::

        H(Y_t | Y_{t-1}, X_t) = -\sum_{y_t, y_{t-1}, x_t} p(y_t, y_{t-1}, x_t)
                                \log p(y_t | y_{t-1}, x_t)

    where:

    .. math::

        p(y_t | y_{t-1}, x_t) = \frac{p(y_t, y_{t-1}, x_t)}{p(y_{t-1}, x_t)}

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution tensor p(y_t, y_{t-1}, x_t).
        Shape: ``(num_y_curr, num_y_prev, num_x_curr)`` or with batch dims
        ``(..., num_y_curr, num_y_prev, num_x_curr)``.

        - Dimension -3 corresponds to current Y state (y_t)
        - Dimension -2 corresponds to previous Y state (y_{t-1})
        - Dimension -1 corresponds to current X state (x_t)

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
        Causally conditioned entropy H(Y||X). Shape depends on reduction:

        - reduction="none": batch dimensions (or scalar if no batch dims)
        - reduction="mean" or "sum": scalar

    Examples
    --------
    >>> # When Y_t is deterministic given (Y_{t-1}, X_t), H(Y||X) = 0
    >>> joint = torch.zeros(2, 2, 2)
    >>> joint[0, 0, 0] = joint[0, 0, 1] = 0.125  # y_t=0 when y_{t-1}=0
    >>> joint[1, 1, 0] = joint[1, 1, 1] = 0.125  # y_t=1 when y_{t-1}=1
    >>> joint[0, 1, 0] = joint[0, 1, 1] = 0.125  # more...
    >>> joint[1, 0, 0] = joint[1, 0, 1] = 0.125
    >>> causally_conditioned_entropy(joint)  # Should be 0

    >>> # Uniform distribution gives maximum conditional entropy
    >>> joint = torch.ones(2, 2, 2) / 8
    >>> causally_conditioned_entropy(joint)  # Should be log(2) nats

    Notes
    -----
    - H(Y||X) >= 0 always (non-negative)
    - H(Y||X) <= H(Y) (conditioning reduces entropy)
    - H(Y||X) = 0 when Y_t is deterministic given (Y_{t-1}, X_t)
    - Causally conditioned entropy is used in directed information theory
      and measures the intrinsic randomness of Y given causal knowledge of X
    - Supports first-order gradients

    See Also
    --------
    conditional_entropy : Standard conditional entropy H(Y|X).
    transfer_entropy : Information transfer T(X -> Y).
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

    if base is not None and base == 1.0:
        raise ValueError(f"base must not be 1, got {base}")

    return torch.ops.torchscience.causally_conditioned_entropy(
        joint,
        input_type,
        reduction,
        base,
    )
