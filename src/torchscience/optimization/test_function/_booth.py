import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def booth(x1: Tensor, x2: Tensor) -> Tensor:
    r"""
    Booth function.

    Evaluates the Booth test function, a simple 2D quadratic optimization
    benchmark.

    Mathematical Definition
    -----------------------
    For input :math:`(x_1, x_2)`:

    .. math::

        f(x_1, x_2) = (x_1 + 2x_2 - 7)^2 + (2x_1 + x_2 - 5)^2

    The global minimum is at :math:`(x_1^*, x_2^*) = (1, 3)` where
    :math:`f(1, 3) = 0`.

    Properties
    ----------
    - The function is convex, continuous, and differentiable everywhere.
    - It is a quadratic function (bowl-shaped) with a single global minimum.

    Typical Search Domain
    ---------------------
    The function is typically evaluated on :math:`x_1, x_2 \in [-10, 10]`.

    Parameters
    ----------
    x1 : Tensor
        First coordinate. Supports arbitrary shape with broadcasting.
    x2 : Tensor
        Second coordinate. Must be broadcastable with ``x1``.

    Returns
    -------
    Tensor
        The Booth function value at each input point. Output shape is the
        broadcast shape of ``x1`` and ``x2``.

    Examples
    --------
    Evaluate at the global minimum (should be 0):

    >>> x1 = torch.tensor(1.0)
    >>> x2 = torch.tensor(3.0)
    >>> booth(x1, x2)
    tensor(0.)

    Evaluate at the origin:

    >>> x1 = torch.tensor(0.0)
    >>> x2 = torch.tensor(0.0)
    >>> booth(x1, x2)
    tensor(74.)

    Batch evaluation:

    >>> x1 = torch.tensor([1.0, 0.0])
    >>> x2 = torch.tensor([3.0, 0.0])
    >>> booth(x1, x2)
    tensor([ 0., 74.])

    With gradient computation:

    >>> x1 = torch.tensor(1.0, requires_grad=True)
    >>> x2 = torch.tensor(3.0, requires_grad=True)
    >>> y = booth(x1, x2)
    >>> y.backward()
    >>> x1.grad
    tensor(0.)
    >>> x2.grad
    tensor(0.)

    References
    ----------
    - Booth, A.D. "An optimisation technique applied to experimental data."
    """
    return torch.ops.torchscience.booth(x1, x2)
