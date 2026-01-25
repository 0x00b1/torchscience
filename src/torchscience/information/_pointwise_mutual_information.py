"""Pointwise Mutual Information (PMI) operator."""

from typing import Literal

import torch
from torch import Tensor


def pointwise_mutual_information(
    joint: Tensor,
    dims: tuple[int, int] = (-2, -1),
    *,
    input_type: Literal["probability", "log_probability"] = "probability",
    base: float | None = None,
) -> Tensor:
    r"""Compute pointwise mutual information (PMI) for each element of a joint distribution.

    Pointwise mutual information measures the co-occurrence of two events
    compared to what would be expected if they were independent:

    .. math::
        \text{PMI}(x, y) = \log \frac{p(x, y)}{p(x) \cdot p(y)}

    Unlike mutual information which returns a scalar, PMI returns a tensor
    of the same shape as the input, with each element indicating the
    association strength between the corresponding (x, y) pair.

    - PMI > 0: x and y co-occur more often than expected under independence
    - PMI = 0: x and y co-occur as expected under independence
    - PMI < 0: x and y co-occur less often than expected under independence

    Parameters
    ----------
    joint : Tensor
        Joint probability distribution p(x, y). The tensor should have at least
        2 dimensions. The dimensions specified by ``dims`` are summed over to
        compute the marginal distributions p(x) and p(y).
    dims : tuple[int, int], default=(-2, -1)
        The two dimensions that represent the joint distribution. The first
        dimension corresponds to variable X, the second to variable Y.
    input_type : {"probability", "log_probability"}, default="probability"
        Whether the input represents probabilities or log-probabilities.
    base : float | None, default=None
        The logarithm base. If None, uses natural logarithm. Common choices
        are 2 (bits), e (nats), or 10 (bans/hartleys).

    Returns
    -------
    Tensor
        Pointwise mutual information for each element, same shape as input.

    Notes
    -----
    The marginal distributions are computed by summing over the appropriate
    dimensions:

    - p(x) = sum_y p(x, y)
    - p(y) = sum_x p(x, y)

    For batch dimensions (dimensions not in ``dims``), PMI is computed
    independently for each batch element.

    Relationship with Mutual Information:

    .. math::
        I(X; Y) = \sum_{x, y} p(x, y) \cdot \text{PMI}(x, y)

    That is, mutual information is the expected value of PMI.

    Variants:

    - **PPMI** (Positive PMI): max(0, PMI(x, y))
    - **NPMI** (Normalized PMI): PMI(x, y) / (-log(p(x, y)))

    Examples
    --------
    >>> import torch
    >>> from torchscience.information import pointwise_mutual_information
    >>> # Joint distribution where X and Y are positively correlated
    >>> joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]])  # p(X, Y)
    >>> pmi = pointwise_mutual_information(joint)
    >>> pmi
    tensor([[ 0.4700, -0.9163],
            [-0.9163,  0.4700]])
    >>> # Diagonal elements are positive (co-occur more than expected)
    >>> # Off-diagonal elements are negative (co-occur less than expected)

    >>> # Independent distribution
    >>> p_x = torch.tensor([0.3, 0.7])
    >>> p_y = torch.tensor([0.4, 0.6])
    >>> joint_independent = p_x.unsqueeze(1) * p_y.unsqueeze(0)
    >>> pmi_independent = pointwise_mutual_information(joint_independent)
    >>> pmi_independent
    tensor([[ 0.0000e+00,  0.0000e+00],
            [-5.9605e-08, -5.9605e-08]])
    >>> # All PMI values are (approximately) zero for independent distributions

    See Also
    --------
    mutual_information : Expected value of PMI over the joint distribution.
    joint_entropy : Entropy of the joint distribution.
    """
    return torch.ops.torchscience.pointwise_mutual_information(
        joint, dims, input_type, base
    )
