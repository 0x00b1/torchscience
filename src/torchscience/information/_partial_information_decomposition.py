"""Partial information decomposition operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def partial_information_decomposition(
    joint_xyz: Tensor,
    *,
    method: Literal["imin"] = "imin",
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    base: Optional[float] = None,
) -> dict[str, Tensor]:
    r"""Compute partial information decomposition of joint source (X, Y) about target Z.

    Partial Information Decomposition (PID) decomposes the mutual information
    I(X,Y;Z) between sources (X,Y) and target Z into four non-negative components:

    .. math::

        I(X,Y;Z) = \text{Redundancy} + \text{Unique}(X) + \text{Unique}(Y) + \text{Synergy}

    where:

    - **Redundancy**: Information that both X and Y provide about Z (shared/overlapping)
    - **Unique(X)**: Information only X provides about Z
    - **Unique(Y)**: Information only Y provides about Z
    - **Synergy**: Information that X and Y together provide that neither provides alone

    The decomposition uses the Williams-Beer Imin measure for redundancy:

    .. math::

        \text{Redundancy} = \sum_z p(z) \cdot \min(I_{spec,X}(z), I_{spec,Y}(z))

    where the specific information is:

    .. math::

        I_{spec,X}(z) = \sum_x p(x|z) \log \frac{p(z|x)}{p(z)}

    Parameters
    ----------
    joint_xyz : Tensor
        Joint probability distribution tensor p(x, y, z).
        Shape: ``(num_x, num_y, num_z)`` or with batch dims
        ``(..., num_x, num_y, num_z)``.

        - Dimension -3 corresponds to source X
        - Dimension -2 corresponds to source Y
        - Dimension -1 corresponds to target Z

    method : {"imin"}, default="imin"
        PID measure to use:

        - ``"imin"``: Williams-Beer Imin measure (minimum specific information)

    input_type : {"probability", "log_probability", "logits"}, default="probability"
        Type of input values:

        - ``"probability"``: Values are probabilities in [0, 1]
        - ``"log_probability"``: Values are log probabilities
        - ``"logits"``: Values are unnormalized log probabilities

    base : float or None, default=None
        Logarithm base. None for natural log (nats), 2 for bits, 10 for dits.

    Returns
    -------
    dict[str, Tensor]
        Dictionary containing:

        - ``"redundancy"``: Shared information R(X,Y;Z)
        - ``"unique_x"``: Information unique to X
        - ``"unique_y"``: Information unique to Y
        - ``"synergy"``: Synergistic information
        - ``"mutual_information"``: Total I(X,Y;Z) = R + U_X + U_Y + S

        Each tensor has shape matching batch dimensions (or scalar if no batch dims).

    Examples
    --------
    >>> # XOR gate: purely synergistic
    >>> joint = torch.zeros(2, 2, 2)
    >>> joint[0, 0, 0] = joint[0, 1, 1] = joint[1, 0, 1] = joint[1, 1, 0] = 0.25
    >>> result = partial_information_decomposition(joint)
    >>> result["synergy"]  # Should be ~log(2) nats

    >>> # COPY gate: X is copied to Z, Y is noise
    >>> joint = torch.zeros(2, 2, 2)
    >>> for y in range(2):
    ...     joint[0, y, 0] = joint[1, y, 1] = 0.25
    >>> result = partial_information_decomposition(joint)
    >>> result["unique_x"]  # Should be ~log(2) nats

    >>> # AND gate: mixture of redundancy and synergy
    >>> joint = torch.zeros(2, 2, 2)
    >>> joint[0, 0, 0] = joint[0, 1, 0] = joint[1, 0, 0] = 0.25
    >>> joint[1, 1, 1] = 0.25  # Z=1 only when X=1 AND Y=1
    >>> result = partial_information_decomposition(joint)

    Notes
    -----
    - All components are non-negative by construction
    - The decomposition satisfies: I(X,Y;Z) = Redundancy + Unique(X) + Unique(Y) + Synergy
    - Williams-Beer Imin is the original PID measure but has known limitations
      (e.g., it can assign zero unique information when intuitively there should be some)
    - Supports first-order gradients using subgradients for the min operation in
      redundancy; note that gradients are approximate due to the complex dependencies
      through marginal distributions

    References
    ----------
    .. [1] Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of
           multivariate information. arXiv:1004.2515.

    See Also
    --------
    mutual_information : Total mutual information I(X;Y).
    conditional_mutual_information : Conditional mutual information I(X;Y|Z).
    """
    if not isinstance(joint_xyz, Tensor):
        raise TypeError(f"joint_xyz must be a Tensor, got {type(joint_xyz)}")

    if joint_xyz.dim() < 3:
        raise ValueError(
            f"joint_xyz must have at least 3 dimensions, got {joint_xyz.dim()}"
        )

    valid_methods = ("imin",)
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )

    valid_input_types = ("probability", "log_probability", "logits")
    if input_type not in valid_input_types:
        raise ValueError(
            f"input_type must be one of {valid_input_types}, got '{input_type}'"
        )

    if base is not None and base <= 0:
        raise ValueError(f"base must be positive, got {base}")

    results = torch.ops.torchscience.partial_information_decomposition(
        joint_xyz,
        method,
        input_type,
        base,
    )

    return {
        "redundancy": results[0],
        "unique_x": results[1],
        "unique_y": results[2],
        "synergy": results[3],
        "mutual_information": results[4],
    }
