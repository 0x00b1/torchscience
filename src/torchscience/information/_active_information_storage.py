"""Active information storage operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def active_information_storage(
    joint: Tensor,
    *,
    input_type: Literal["probability", "log_probability"] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute active information storage A(X) from a joint transition distribution.

    Active information storage measures how much information from the past of a
    process is actively used in computing its next state. It quantifies the
    memory or predictability of a stochastic process:

    .. math::

        A(X) = I(X_{t-1}; X_t)
             = H(X_t) - H(X_t | X_{t-1})
             = H(X_{t-1}) - H(X_{t-1} | X_t)

    This is the mutual information between consecutive time steps of a process.

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution tensor p(x_t, x_{t-1}).
        Shape: ``(num_curr_states, num_prev_states)`` or with batch dims
        ``(..., num_curr_states, num_prev_states)``.

        - Dimension -2 corresponds to current state (x_t)
        - Dimension -1 corresponds to previous state (x_{t-1})

    input_type : {"probability", "log_probability"}, default="probability"
        Type of input values:

        - ``"probability"``: Values are probabilities in [0, 1]
        - ``"log_probability"``: Values are log probabilities

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
        Active information storage A(X). Shape depends on reduction:

        - reduction="none": batch dimensions (or scalar if no batch dims)
        - reduction="mean" or "sum": scalar

    Examples
    --------
    >>> # For an i.i.d. process (no memory), A(X) = 0
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> joint = p_x.unsqueeze(0) * p_x.unsqueeze(1)  # Independent
    >>> active_information_storage(joint)
    tensor(0.)

    >>> # For a deterministic process X_t = X_{t-1}, A(X) = H(X)
    >>> joint = torch.tensor([[0.5, 0.0], [0.0, 0.5]])  # Identity transition
    >>> active_information_storage(joint, base=2.0)
    tensor(1.)  # 1 bit, since H(X) = log2(2) = 1

    >>> # For a Markov chain with memory
    >>> joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]])
    >>> active_information_storage(joint)  # Positive, reflecting memory

    Notes
    -----
    - A(X) >= 0 always (non-negative, as it's mutual information)
    - A(X) = 0 for an i.i.d. process (X_t independent of X_{t-1})
    - A(X) = H(X) for a deterministic process (past determines future)
    - A(X) is bounded: A(X) <= min(H(X_t), H(X_{t-1}))
    - Also known as "excess entropy" or "predictive information" in some contexts
    - Supports first-order gradients

    See Also
    --------
    mutual_information : General mutual information I(X;Y).
    transfer_entropy : Directed information transfer T(X -> Y).
    conditional_entropy : Conditional entropy H(X|Y).
    """
    if not isinstance(joint, Tensor):
        raise TypeError(f"joint must be a Tensor, got {type(joint)}")

    if joint.dim() < 2:
        raise ValueError(
            f"joint must have at least 2 dimensions, got {joint.dim()}"
        )

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

    if base is not None and base <= 0:
        raise ValueError(f"base must be positive, got {base}")

    if base is not None and base == 1.0:
        raise ValueError("base must not be 1.0")

    return torch.ops.torchscience.active_information_storage(
        joint,
        input_type,
        reduction,
        base,
    )
