import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def beale(x1: Tensor, x2: Tensor) -> Tensor:
    r"""
    Beale function.

    Evaluates the Beale test function, a multimodal 2D optimization benchmark
    with sharp peaks and a narrow valley.

    Mathematical Definition
    -----------------------
    For input :math:`(x_1, x_2)`:

    .. math::

        f(x_1, x_2) = (1.5 - x_1 + x_1 x_2)^2
                     + (2.25 - x_1 + x_1 x_2^2)^2
                     + (2.625 - x_1 + x_1 x_2^3)^2

    The global minimum is at :math:`(x_1^*, x_2^*) = (3, 0.5)` where
    :math:`f(3, 0.5) = 0`.

    Properties
    ----------
    - The function is continuous and differentiable everywhere.
    - It has a flat region near the origin and steep walls.
    - Higher-order polynomial terms make it challenging for optimization.

    Typical Search Domain
    ---------------------
    The function is typically evaluated on :math:`x_1, x_2 \in [-4.5, 4.5]`.

    Parameters
    ----------
    x1 : Tensor
        First coordinate. Supports arbitrary shape with broadcasting.
    x2 : Tensor
        Second coordinate. Must be broadcastable with ``x1``.

    Returns
    -------
    Tensor
        The Beale function value at each input point. Output shape is the
        broadcast shape of ``x1`` and ``x2``.

    Examples
    --------
    Evaluate at the global minimum (should be 0):

    >>> x1 = torch.tensor(3.0)
    >>> x2 = torch.tensor(0.5)
    >>> beale(x1, x2)
    tensor(0.)

    References
    ----------
    - Beale, E.M.L. "On an iterative method for finding a local minimum of a
      function of more than one variable." Technical Report 25, Statistical
      Techniques Research Group, Princeton University (1958).
    """
    return torch.ops.torchscience.beale(x1, x2)
