import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def ackley(x: Tensor) -> Tensor:
    r"""
    Ackley function.

    Evaluates the Ackley test function, a widely used multimodal optimization
    benchmark characterized by a nearly flat outer region and a large hole
    at the center.

    Mathematical Definition
    -----------------------
    For an n-dimensional input :math:`\mathbf{x} = (x_1, \ldots, x_n)`:

    .. math::

        f(\mathbf{x}) = -20 \exp\!\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}\right)
                       - \exp\!\left(\frac{1}{n}\sum_{i=1}^n \cos(2\pi x_i)\right)
                       + 20 + e

    The global minimum is at the origin :math:`\mathbf{x}^* = \mathbf{0}`
    where :math:`f(\mathbf{0}) = 0`.

    Properties
    ----------
    - The function has a nearly flat outer region that drops sharply near
      the origin, making it challenging for hill-climbing algorithms.
    - It is multimodal with many local minima.
    - The function is non-separable due to the square root of the mean.

    Typical Search Domain
    ---------------------
    The function is typically evaluated on :math:`x_i \in [-5, 5]`.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(..., n)`` where ``n >= 1`` is the dimension
        of the optimization problem. The last dimension contains the coordinates
        of each point. Batch dimensions are fully supported.

    Returns
    -------
    Tensor
        The Ackley function value at each input point. Output shape is
        ``x.shape[:-1]`` (the last dimension is reduced).

    Examples
    --------
    Evaluate at the global minimum (should be 0):

    >>> x = torch.zeros(3)
    >>> ackley(x)
    tensor(0.)

    Batch evaluation:

    >>> x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> ackley(x)
    tensor([0.0000, 3.6254])

    References
    ----------
    - Ackley, D.H. "A connectionist machine for genetic hillclimbing."
      Kluwer Academic Publishers (1987).
    """
    return torch.ops.torchscience.ackley(x)
