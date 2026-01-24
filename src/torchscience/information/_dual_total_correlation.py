"""Dual total correlation operator."""

from typing import Literal, Optional

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - loads C++ extension


def dual_total_correlation(
    joint: Tensor,
    *,
    input_type: Literal[
        "probability", "log_probability", "logits"
    ] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute dual total correlation (binding information) from a joint distribution.

    The dual total correlation measures the shared information that binds all
    variables together:

    .. math::

        DTC(X_1, \ldots, X_n) = H(X_1, \ldots, X_n) - \sum_{i=1}^{n} H(X_i | X_{-i})

    Where :math:`H(X_i | X_{-i})` is the entropy of :math:`X_i` conditioned on all
    other variables :math:`X_{-i} = (X_1, \ldots, X_{i-1}, X_{i+1}, \ldots, X_n)`.

    Unlike total correlation which measures redundancy, dual total correlation
    measures how much knowing all-but-one variable tells you about the remaining one.

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
        Dual total correlation value. Shape depends on reduction and whether there
        are batch dimensions.

    Examples
    --------
    >>> # Independent variables (product distribution)
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> p_y = torch.tensor([0.5, 0.5])
    >>> joint_indep = torch.outer(p_x, p_y)  # Product distribution
    >>> dual_total_correlation(joint_indep)  # Should be ~0
    tensor(0.)

    >>> # Perfectly correlated variables
    >>> joint_corr = torch.zeros(2, 2)
    >>> joint_corr[0, 0] = joint_corr[1, 1] = 0.5
    >>> dual_total_correlation(joint_corr)  # Should be log(2) ~ 0.693
    tensor(0.6931)

    Notes
    -----
    - DTC >= 0 always (non-negative)
    - DTC = 0 iff all variables are mutually independent
    - For n=2 variables, DTC equals mutual information I(X;Y), same as TC
    - Also known as binding information or excess entropy
    - Supports first-order gradients

    See Also
    --------
    mutual_information : Mutual information for two variables.
    total_correlation : Alternative multivariate dependence measure.
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

    return torch.ops.torchscience.dual_total_correlation(
        joint,
        input_type,
        reduction,
        base,
    )
