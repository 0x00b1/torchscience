"""Directed information operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def directed_information(
    joint: Tensor,
    *,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute directed information I(X -> Y) from a joint distribution.

    Directed information measures the causal information flow from process X
    to process Y at the current time step, conditioned on the past of Y:

    .. math::

        I(X \to Y) = I(X_t; Y_t | Y_{t-1})
                   = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_t)

    This can be computed as:

    .. math::

        I(X \to Y) = \sum_{y_t, y_{t-1}, x_t} p(y_t, y_{t-1}, x_t)
                     \log \frac{p(y_t | y_{t-1}, x_t)}{p(y_t | y_{t-1})}

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
        Directed information I(X -> Y). Shape depends on reduction:

        - reduction="none": batch dimensions (or scalar if no batch dims)
        - reduction="mean" or "sum": scalar

    Examples
    --------
    >>> # Independent processes - no information transfer
    >>> joint = torch.ones(2, 2, 2) / 8  # Uniform distribution
    >>> directed_information(joint)  # Should be ~0

    >>> # X perfectly determines Y_t (given Y_{t-1})
    >>> joint = torch.zeros(2, 2, 2)
    >>> joint[0, 0, 0] = joint[0, 1, 0] = 0.25  # When x_t=0, y_t=0
    >>> joint[1, 0, 1] = joint[1, 1, 1] = 0.25  # When x_t=1, y_t=1
    >>> directed_information(joint)  # Should be log(2) nats

    Notes
    -----
    - I(X -> Y) >= 0 always (non-negative)
    - I(X -> Y) = 0 when X provides no information about Y given Y's past
    - I(X -> Y) != I(Y -> X) in general (asymmetric measure)
    - Directed information is closely related to transfer entropy. The key
      difference is that transfer entropy uses X_{t-1} (past of X), while
      directed information uses X_t (current X).
    - Directed information is a special case of conditional mutual information:
      I(X -> Y) = I(X_t; Y_t | Y_{t-1})
    - Supports first-order gradients

    See Also
    --------
    transfer_entropy : Transfer entropy T(X -> Y) = I(Y_t; X_{t-1} | Y_{t-1}).
    conditional_mutual_information : General conditional mutual information I(X;Y|Z).
    mutual_information : Unconditional mutual information I(X;Y).
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

    return torch.ops.torchscience.directed_information(
        joint,
        input_type,
        reduction,
        base,
    )
