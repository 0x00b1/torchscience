"""Mutual information implementation."""

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def mutual_information(
    joint: Tensor,
    *,
    dims: Tuple[int, int] = (-2, -1),
    input_type: Literal["probability", "log_probability"] = "probability",
    reduction: Literal["none", "mean", "sum"] = "none",
    base: Optional[float] = None,
) -> Tensor:
    r"""Compute mutual information I(X;Y) from joint probability distribution.

    Mutual information measures the statistical dependence between two random
    variables - how much knowing one variable reduces uncertainty about the other.

    Mathematical Definition
    -----------------------
    For a joint distribution P(X, Y):

    .. math::
        I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
                = H(X) + H(Y) - H(X, Y)
                = H(Y) - H(Y|X)
                = H(X) - H(X|Y)

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution P(X, Y). The dimensions specified by
        ``dims`` are summed over to compute the mutual information.
    dims : Tuple[int, int], default=(-2, -1)
        The two dimensions indexing X and Y in the joint distribution.
    input_type : {"probability", "log_probability"}, default="probability"
        How to interpret input tensors:

        - ``"probability"``: Direct probability values (epsilon-clamped)
        - ``"log_probability"``: Log-probabilities (exponentiated before use)
    reduction : {"none", "mean", "sum"}, default="none"
        Reduction to apply:

        - ``"none"``: Return per-sample mutual information values
        - ``"mean"``: Mean over batch dimensions
        - ``"sum"``: Sum over batch dimensions
    base : float or None, default=None
        Logarithm base for mutual information calculation:

        - ``None``: Natural logarithm (MI in nats)
        - ``2``: Base-2 logarithm (MI in bits)
        - ``10``: Base-10 logarithm

    Returns
    -------
    Tensor
        Mutual information values I(X;Y). Shape depends on ``reduction``.

    Examples
    --------
    >>> # For independent variables, I(X;Y) = 0
    >>> p_x = torch.tensor([0.5, 0.5])
    >>> p_y = torch.tensor([0.25, 0.25, 0.25, 0.25])
    >>> joint = p_x.unsqueeze(1) * p_y.unsqueeze(0)  # Independent
    >>> torchscience.information_theory.mutual_information(joint)
    tensor(0.0)

    >>> # For perfectly correlated variables, I(X;Y) = H(X) = H(Y)
    >>> joint = torch.tensor([[0.5, 0.0], [0.0, 0.5]])  # Y = X
    >>> torchscience.information_theory.mutual_information(joint, base=2.0)
    tensor(1.0)  # 1 bit of information

    Notes
    -----
    - Mutual information is symmetric: :math:`I(X;Y) = I(Y;X)`
    - Non-negative: :math:`I(X;Y) \geq 0`
    - Zero for independent variables: :math:`I(X;Y) = 0` iff X ⊥ Y
    - Bounded: :math:`I(X;Y) \leq \min(H(X), H(Y))`
    - Supports first and second-order gradients

    See Also
    --------
    joint_entropy : Joint entropy H(X, Y).
    conditional_entropy : Conditional entropy H(Y|X).
    shannon_entropy : Marginal entropy H(X).
    """
    if not isinstance(joint, Tensor):
        raise TypeError(f"joint must be a Tensor, got {type(joint).__name__}")

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

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"base must be positive and not equal to 1, got {base}"
        )

    if len(dims) != 2:
        raise ValueError(f"dims must have exactly 2 elements, got {len(dims)}")

    ndim = joint.dim()
    if ndim < 2:
        raise ValueError("joint must have at least 2 dimensions")

    # Normalize and validate dims
    dim0 = dims[0] if dims[0] >= 0 else ndim + dims[0]
    dim1 = dims[1] if dims[1] >= 0 else ndim + dims[1]

    if dim0 < 0 or dim0 >= ndim:
        raise IndexError(
            f"dims[0]={dims[0]} out of range for tensor with {ndim} dimensions"
        )
    if dim1 < 0 or dim1 >= ndim:
        raise IndexError(
            f"dims[1]={dims[1]} out of range for tensor with {ndim} dimensions"
        )
    if dim0 == dim1:
        raise ValueError("dims must contain two different dimensions")

    return torch.ops.torchscience.mutual_information(
        joint, list(dims), input_type, reduction, base
    )
