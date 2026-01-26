import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def sphere(x: Tensor) -> Tensor:
    r"""
    Sphere function.

    Evaluates the sphere test function at each point in the input tensor.
    This is one of the simplest continuous optimization test functions,
    commonly used as a baseline for benchmarking optimization algorithms.

    Mathematical Definition
    -----------------------
    For an n-dimensional input :math:`\mathbf{x} = (x_1, \ldots, x_n)`:

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^n x_i^2

    The global minimum is at the origin :math:`\mathbf{x}^* = \mathbf{0}`
    where :math:`f(\mathbf{0}) = 0`.

    Properties
    ----------
    - The function is convex, continuous, and differentiable everywhere.
    - It is unimodal with a single global minimum.
    - The gradient is :math:`\nabla f(\mathbf{x}) = 2\mathbf{x}`.
    - The Hessian is :math:`2\mathbf{I}`, making it perfectly conditioned.

    Typical Search Domain
    ---------------------
    The function is typically evaluated on :math:`x_i \in [-5.12, 5.12]`.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(..., n)`` where ``n >= 1`` is the dimension
        of the optimization problem. The last dimension contains the coordinates
        of each point. Batch dimensions are fully supported.

    Returns
    -------
    Tensor
        The sphere function value at each input point. Output shape is
        ``x.shape[:-1]`` (the last dimension is reduced). For a single point
        input of shape ``(n,)``, returns a scalar tensor.

    Examples
    --------
    Evaluate at the global minimum (should be 0):

    >>> x = torch.zeros(3)
    >>> sphere(x)
    tensor(0.)

    Evaluate at a non-optimal point:

    >>> x = torch.tensor([1.0, 2.0, 3.0])
    >>> sphere(x)
    tensor(14.)

    Batch evaluation:

    >>> x = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    >>> sphere(x)
    tensor([1., 0., 2.])

    With gradient computation:

    >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = sphere(x)
    >>> y.backward()
    >>> x.grad
    tensor([2., 4., 6.])

    References
    ----------
    - Molga, M., Smutnicki, C. "Test functions for optimization needs."
      (2005).
    """
    return torch.ops.torchscience.sphere(x)
