"""O-information operator."""

from typing import Literal, Optional

from torch import Tensor

from ._dual_total_correlation import dual_total_correlation
from ._total_correlation import total_correlation


def o_information(
    joint: Tensor,
    *,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute O-information from a joint distribution.

    O-information measures the balance between redundancy and synergy in a
    multivariate system:

    .. math::

        \Omega(X_1, \ldots, X_n) = TC(X_1, \ldots, X_n) - DTC(X_1, \ldots, X_n)

    Where TC is total correlation and DTC is dual total correlation.

    The sign of O-information indicates the dominant type of interaction:

    - :math:`\Omega > 0`: Redundancy-dominated (variables share overlapping information)
    - :math:`\Omega < 0`: Synergy-dominated (variables together convey more than separately)
    - :math:`\Omega = 0`: Perfect balance between redundancy and synergy

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution tensor p(x_1, ..., x_n). The tensor
        should have at least 2 dimensions, where each dimension corresponds
        to a random variable.
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
        O-information value. Shape depends on reduction and whether there
        are batch dimensions.

    Examples
    --------
    >>> # Independent variables: O-information = 0
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> p_y = torch.tensor([0.5, 0.5])
    >>> joint_indep = torch.outer(p_x, p_y)
    >>> o_information(joint_indep)  # Should be ~0
    tensor(0.)

    >>> # For bivariate case: O-information = 0 (TC = DTC = I(X;Y))
    >>> joint = torch.rand(3, 3)
    >>> joint = joint / joint.sum()
    >>> o_information(joint)  # Should be ~0
    tensor(0.)

    Notes
    -----
    - For n=2 variables, O-information = 0 since TC = DTC = I(X;Y)
    - O-information can be positive (redundancy) or negative (synergy)
    - Also known as the "informational character" of a system
    - Supports first-order gradients through TC and DTC

    See Also
    --------
    total_correlation : Measures redundancy among all variables.
    dual_total_correlation : Measures binding information among variables.
    """
    if not isinstance(joint, Tensor):
        raise TypeError(f"joint must be a Tensor, got {type(joint)}")

    if joint.dim() < 2:
        raise ValueError(
            f"joint must have at least 2 dimensions, got {joint.dim()}"
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

    tc = total_correlation(
        joint, input_type=input_type, reduction=reduction, base=base
    )
    dtc = dual_total_correlation(
        joint, input_type=input_type, reduction=reduction, base=base
    )

    return tc - dtc
