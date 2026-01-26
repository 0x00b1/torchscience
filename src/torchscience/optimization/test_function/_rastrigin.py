import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def rastrigin(x: Tensor) -> Tensor:
    r"""
    Rastrigin function.

    Evaluates the Rastrigin test function, a highly multimodal optimization
    benchmark with regularly distributed local minima.

    Mathematical Definition
    -----------------------
    For an n-dimensional input :math:`\mathbf{x} = (x_1, \ldots, x_n)`:

    .. math::

        f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[ x_i^2 - 10\cos(2\pi x_i) \right]

    The global minimum is at the origin :math:`\mathbf{x}^* = \mathbf{0}`
    where :math:`f(\mathbf{0}) = 0`.

    Properties
    ----------
    - The function is highly multimodal with approximately :math:`11^n` local
      minima for the typical search domain.
    - The cosine term produces regularly spaced local minima.
    - It is a non-convex function that is challenging for gradient-based
      optimization.
    - The function is separable: each variable contributes independently.

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
        The Rastrigin function value at each input point. Output shape is
        ``x.shape[:-1]`` (the last dimension is reduced).

    Examples
    --------
    Evaluate at the global minimum (should be 0):

    >>> x = torch.zeros(3)
    >>> rastrigin(x)
    tensor(0.)

    Evaluate at a non-optimal point:

    >>> x = torch.ones(3)
    >>> rastrigin(x)
    tensor(3.)

    Batch evaluation:

    >>> x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    >>> rastrigin(x)
    tensor([0., 2.])

    References
    ----------
    - Rastrigin, L.A. "Systems of extremal control." Nauka (1974).
    - Muhlenbein, H., Schomisch, D., Born, J. "The parallel genetic algorithm
      as function optimizer." Parallel Computing 17.6-7 (1991): 619-632.
    """
    return torch.ops.torchscience.rastrigin(x)
