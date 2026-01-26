import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def himmelblau(x1: Tensor, x2: Tensor) -> Tensor:
    r"""
    Himmelblau function.

    Evaluates the Himmelblau test function, a multimodal 2D optimization
    benchmark with four identical global minima.

    Mathematical Definition
    -----------------------
    For input :math:`(x_1, x_2)`:

    .. math::

        f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2

    The function has four global minima, all with :math:`f = 0`:

    - :math:`(3.0, 2.0)`
    - :math:`(-2.805118, 3.131312)`
    - :math:`(-3.779310, -3.283186)`
    - :math:`(3.584428, -1.848126)`

    Properties
    ----------
    - The function is continuous and differentiable everywhere.
    - It has four identical global minima, making it useful for testing
      multimodal optimization algorithms.
    - There is one local maximum at approximately :math:`(-0.270845, -0.923039)`
      with :math:`f \approx 181.617`.

    Typical Search Domain
    ---------------------
    The function is typically evaluated on :math:`x_1, x_2 \in [-5, 5]`.

    Parameters
    ----------
    x1 : Tensor
        First coordinate. Supports arbitrary shape with broadcasting.
    x2 : Tensor
        Second coordinate. Must be broadcastable with ``x1``.

    Returns
    -------
    Tensor
        The Himmelblau function value at each input point. Output shape is
        the broadcast shape of ``x1`` and ``x2``.

    Examples
    --------
    Evaluate at one of the global minima (should be 0):

    >>> x1 = torch.tensor(3.0)
    >>> x2 = torch.tensor(2.0)
    >>> himmelblau(x1, x2)
    tensor(0.)

    References
    ----------
    - Himmelblau, D. "Applied Nonlinear Programming." McGraw-Hill (1972).
    """
    return torch.ops.torchscience.himmelblau(x1, x2)
