"""Interaction information operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def interaction_information(
    joint: Tensor,
    *,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute interaction information I(X;Y;Z) from a joint distribution.

    The interaction information measures the amount of information (positive or
    negative) that the variable Z provides about the relationship between X and Y:

    .. math::

        I(X;Y;Z) = I(X;Y) - I(X;Y|Z)

    This is equivalent to the entropy-based formula:

    .. math::

        I(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z)

    Unlike mutual information, interaction information can be negative (synergy)
    or positive (redundancy):

    - **Positive (redundancy)**: The variables share overlapping information about
      each other. Knowing Z reduces the information that X and Y share.
    - **Negative (synergy)**: The variables together convey more information than
      separately. Knowing Z increases the information that X and Y share.

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution tensor p(x,y,z) with exactly 3 dimensions,
        where dimensions 0, 1, 2 correspond to variables X, Y, Z respectively.
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
        Interaction information value I(X;Y;Z). Can be positive (redundancy)
        or negative (synergy).

    Examples
    --------
    >>> # Independent variables: I(X;Y;Z) = 0
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> p_y = torch.tensor([0.5, 0.5])
    >>> p_z = torch.tensor([0.5, 0.5])
    >>> joint_indep = torch.einsum("i,j,k->ijk", p_x, p_y, p_z)
    >>> interaction_information(joint_indep)  # Should be ~0
    tensor(0.)

    >>> # Redundant case: X = Y = Z (all identical)
    >>> joint_redundant = torch.zeros(2, 2, 2)
    >>> joint_redundant[0, 0, 0] = joint_redundant[1, 1, 1] = 0.5
    >>> interaction_information(joint_redundant)  # Should be > 0
    tensor(0.6931)

    Notes
    -----
    - I(X;Y;Z) can be positive (redundancy) or negative (synergy)
    - I(X;Y;Z) = 0 when variables are mutually independent
    - Also known as co-information (with opposite sign convention in some references)
    - Symmetric in all three variables: I(X;Y;Z) = I(Y;X;Z) = I(Z;X;Y)
    - Supports first-order gradients

    See Also
    --------
    mutual_information : Mutual information for two variables.
    conditional_mutual_information : Conditional mutual information I(X;Y|Z).
    total_correlation : Total correlation (multi-information).
    """
    if not isinstance(joint, Tensor):
        raise TypeError(f"joint must be a Tensor, got {type(joint)}")

    if joint.dim() != 3:
        raise ValueError(
            f"joint must have exactly 3 dimensions, got {joint.dim()}"
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

    return torch.ops.torchscience.interaction_information(
        joint,
        input_type,
        reduction,
        base,
    )
