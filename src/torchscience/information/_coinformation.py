"""Coinformation operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def coinformation(
    joint: Tensor,
    *,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute coinformation from an N-dimensional joint distribution.

    Coinformation (also known as multi-information or interaction information
    in the general case) is the generalization of mutual information to N
    variables using the inclusion-exclusion principle on entropies:

    .. math::

        CI(X_1, \ldots, X_n) = -\sum_{S \subseteq \{1,\ldots,n\}, S \neq \emptyset}
                               (-1)^{|S|} H(X_S)

    Or equivalently using the alternating sign pattern:

    .. math::

        CI = \sum_{k=1}^{n} (-1)^{k+1} \sum_{|S|=k} H(X_S)

    Special cases:
        - For n=2: :math:`CI(X;Y) = H(X) + H(Y) - H(X,Y) = I(X;Y)` (mutual information)
        - For n=3: :math:`CI(X;Y;Z) = H(X) + H(Y) + H(Z) - H(X,Y) - H(X,Z) - H(Y,Z) + H(X,Y,Z) = I(X;Y;Z)` (interaction information)

    Coinformation can be positive (redundancy), negative (synergy), or zero
    (independence).

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution tensor p(x_1, ..., x_n). The tensor
        should have at least 2 dimensions and at most 10 dimensions, where
        each dimension corresponds to a random variable.
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
        Coinformation value. Can be positive (redundancy), negative (synergy),
        or zero (independence).

    Examples
    --------
    >>> # Independent 2D case: CI = I(X;Y) = 0
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> p_y = torch.tensor([0.5, 0.5])
    >>> joint_2d = torch.outer(p_x, p_y)
    >>> coinformation(joint_2d)  # Should be ~0
    tensor(0.)

    >>> # Perfectly correlated 2D case: CI = I(X;Y) = log(2)
    >>> joint_corr = torch.zeros(2, 2)
    >>> joint_corr[0, 0] = joint_corr[1, 1] = 0.5
    >>> coinformation(joint_corr)  # Should be log(2)
    tensor(0.6931)

    >>> # 3D case: same as interaction information
    >>> joint_3d = torch.rand(3, 3, 3)
    >>> joint_3d = joint_3d / joint_3d.sum()
    >>> ci_3d = coinformation(joint_3d)

    Notes
    -----
    - CI can be positive (redundancy), negative (synergy), or zero
    - CI = 0 when all variables are mutually independent
    - For n=2, CI equals mutual information I(X;Y)
    - For n=3, CI equals interaction information I(X;Y;Z)
    - Symmetric in all variables
    - Supports first-order gradients
    - Limited to at most 10 dimensions (2^10 - 1 = 1023 subsets)

    See Also
    --------
    mutual_information : Mutual information for two variables.
    interaction_information : Interaction information for three variables.
    total_correlation : Total correlation (multi-information).
    dual_total_correlation : Dual total correlation (binding information).
    """
    if not isinstance(joint, Tensor):
        raise TypeError(f"joint must be a Tensor, got {type(joint)}")

    if joint.dim() < 2:
        raise ValueError(
            f"joint must have at least 2 dimensions, got {joint.dim()}"
        )

    if joint.dim() > 10:
        raise ValueError(
            f"joint must have at most 10 dimensions, got {joint.dim()}"
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

    return torch.ops.torchscience.coinformation(
        joint,
        input_type,
        reduction,
        base,
    )
