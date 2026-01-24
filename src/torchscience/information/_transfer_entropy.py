"""Transfer entropy operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def transfer_entropy(
    joint: Tensor,
    *,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute transfer entropy T(X -> Y) from a joint transition distribution.

    Transfer entropy measures the amount of directed information transfer from
    process X to process Y - how much knowing the past of X reduces uncertainty
    about the future of Y, beyond what the past of Y already tells us:

    .. math::

        T_{X \to Y} = I(Y_t; X_{t-1} | Y_{t-1})
                    = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

    This can be computed as:

    .. math::

        T_{X \to Y} = \sum_{y_t, y_{t-1}, x_{t-1}} p(y_t, y_{t-1}, x_{t-1})
                      \log \frac{p(y_t | y_{t-1}, x_{t-1})}{p(y_t | y_{t-1})}

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution tensor p(y_t, y_{t-1}, x_{t-1}).
        Shape: ``(num_y_curr, num_y_prev, num_x_prev)`` or with batch dims
        ``(..., num_y_curr, num_y_prev, num_x_prev)``.

        - Dimension -3 corresponds to current Y state (y_t)
        - Dimension -2 corresponds to previous Y state (y_{t-1})
        - Dimension -1 corresponds to previous X state (x_{t-1})

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
        Transfer entropy T(X -> Y). Shape depends on reduction:

        - reduction="none": batch dimensions (or scalar if no batch dims)
        - reduction="mean" or "sum": scalar

    Examples
    --------
    >>> # Independent processes - no information transfer
    >>> joint = torch.ones(2, 2, 2) / 8  # Uniform distribution
    >>> transfer_entropy(joint)  # Should be ~0

    >>> # X perfectly predicts Y_t (given Y_{t-1})
    >>> joint = torch.zeros(2, 2, 2)
    >>> joint[0, 0, 0] = joint[0, 1, 0] = 0.25  # When x_{t-1}=0, y_t=0
    >>> joint[1, 0, 1] = joint[1, 1, 1] = 0.25  # When x_{t-1}=1, y_t=1
    >>> transfer_entropy(joint)  # Should be log(2) nats

    Notes
    -----
    - T(X -> Y) >= 0 always (non-negative)
    - T(X -> Y) = 0 when X provides no additional information about Y beyond Y's past
    - T(X -> Y) != T(Y -> X) in general (asymmetric measure)
    - Transfer entropy is a special case of conditional mutual information:
      T(X -> Y) = I(Y_t; X_{t-1} | Y_{t-1})
    - Supports first-order gradients

    See Also
    --------
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

    return torch.ops.torchscience.transfer_entropy(
        joint,
        input_type,
        reduction,
        base,
    )
