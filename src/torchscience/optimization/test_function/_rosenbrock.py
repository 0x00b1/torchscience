import warnings
from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators

# Threshold for numerical stability warning
# For b > 1e6 with float32, b * (x_{i+1} - x_i^2)^2 can overflow
_LARGE_B_THRESHOLD_FLOAT32 = 1e6
_LARGE_B_THRESHOLD_FLOAT16 = 1e3


def rosenbrock(
    x: Tensor,
    *,
    a: Union[float, Tensor] = 1.0,
    b: Union[float, Tensor] = 100.0,
) -> Tensor:
    r"""
    Rosenbrock function.

    Evaluates the Rosenbrock function (also known as Rosenbrock's valley or
    Rosenbrock's banana function) at each point in the input tensor. This is
    a classic non-convex test function for optimization algorithms.

    Mathematical Definition
    -----------------------
    For a 2-dimensional input :math:`\mathbf{x} = (x_1, x_2)`:

    .. math::

        f(x_1, x_2) = (a - x_1)^2 + b(x_2 - x_1^2)^2

    For an n-dimensional input :math:`\mathbf{x} = (x_1, \ldots, x_n)`:

    .. math::

        f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b(x_{i+1} - x_i^2)^2 \right]

    The global minimum is at :math:`\mathbf{x}^* = (a, a^2, a^4, \ldots, a^{2^{n-1}})`
    where :math:`f(\mathbf{x}^*) = 0`. For the standard parameters :math:`a=1, b=100`,
    the global minimum is at :math:`\mathbf{x}^* = (1, 1, \ldots, 1)`.

    Properties
    ----------
    - The function has a narrow, curved valley that makes it challenging for
      many optimization algorithms.
    - For :math:`n \geq 4`, the function has a local minimum near
      :math:`(-1, 1, 1, \ldots, 1)` for the standard parameters.
    - The function is unimodal for :math:`n = 2` with the standard parameters.
    - The Hessian at the minimum is ill-conditioned when :math:`b` is large.
      The condition number is approximately :math:`4b` (about 400 for default parameters).

    Typical Search Domain
    ---------------------
    The function is typically evaluated on :math:`x_i \in [-5, 10]` or
    :math:`x_i \in [-2.048, 2.048]`.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape ``(..., n)`` where ``n >= 2`` is the dimension
        of the optimization problem. The last dimension contains the coordinates
        of each point. Batch dimensions are fully supported.
    a : float or Tensor, optional
        First parameter of the Rosenbrock function. Default is 1.0.
        The global minimum x-coordinate is at ``x_1 = a``.
        Can be a scalar or a tensor that broadcasts with x.
    b : float or Tensor, optional
        Second parameter of the Rosenbrock function. Default is 100.0.
        Controls the steepness of the valley walls. Larger values make the
        optimization problem harder.
        Can be a scalar or a tensor that broadcasts with x.

    Returns
    -------
    Tensor
        The Rosenbrock function value at each input point. Output shape is
        ``x.shape[:-1]`` (the last dimension is reduced). For a single point
        input of shape ``(n,)``, returns a scalar tensor.

    Examples
    --------
    Evaluate at the global minimum (should be 0):

    >>> x = torch.tensor([1.0, 1.0])
    >>> rosenbrock(x)
    tensor(0.)

    Evaluate at a non-optimal point:

    >>> x = torch.tensor([0.0, 0.0])
    >>> rosenbrock(x)
    tensor(1.)

    Batch evaluation:

    >>> x = torch.tensor([[1.0, 1.0], [0.0, 0.0], [-1.0, 1.0]])
    >>> rosenbrock(x)
    tensor([0., 1., 4.])

    Higher-dimensional input:

    >>> x = torch.tensor([1.0, 1.0, 1.0, 1.0])
    >>> rosenbrock(x)
    tensor(0.)

    With gradient computation:

    >>> x = torch.tensor([0.0, 0.0], requires_grad=True)
    >>> y = rosenbrock(x)
    >>> y.backward()
    >>> x.grad
    tensor([-2., 0.])

    Custom scalar parameters:

    >>> x = torch.tensor([2.0, 4.0])
    >>> rosenbrock(x, a=2.0, b=50.0)
    tensor(0.)

    Tensor parameters (different a for each batch element):

    >>> x = torch.tensor([[1.0, 1.0], [2.0, 4.0]])
    >>> a = torch.tensor([1.0, 2.0])
    >>> rosenbrock(x, a=a.unsqueeze(-1))
    tensor([0., 0.])

    References
    ----------
    - Rosenbrock, H.H. "An automatic method for finding the greatest or least
      value of a function." The Computer Journal 3.3 (1960): 175-184.

    Notes
    -----
    - The input must have at least 2 elements in the last dimension (n >= 2).
    - For gradient-based optimization, consider that the gradient is:
      :math:`\\frac{\\partial f}{\\partial x_i} = -2(a - x_i) - 4bx_i(x_{i+1} - x_i^2)`
      for :math:`i < n`, with additional terms from the :math:`(i-1)`-th component.
    - The function supports all floating-point dtypes and complex dtypes.
    - When a or b are tensors, they must be broadcastable with x[..., :-1].

    Warnings
    --------
    For large values of ``b`` combined with low-precision dtypes, numerical
    overflow may occur. A warning is issued when:

    - ``b > 1e6`` with float32
    - ``b > 1e3`` with float16 or bfloat16

    Consider using float64 for better numerical stability with large ``b``.
    """
    # For quantized tensors, use float32 for parameters
    param_dtype = x.dtype
    if x.dtype in (
        torch.qint8,
        torch.quint8,
        torch.qint32,
        torch.quint4x2,
        torch.quint2x4,
    ):
        param_dtype = torch.float32

    if not isinstance(a, Tensor):
        a = torch.as_tensor(a, dtype=param_dtype, device=x.device)
    if not isinstance(b, Tensor):
        b = torch.as_tensor(b, dtype=param_dtype, device=x.device)

    # Check for potential numerical instability with large b values
    # Skip check for meta tensors and during torch.compile (can't call .item())
    b_scalar = None
    if b.numel() == 1 and not b.is_meta and not torch.compiler.is_compiling():
        b_scalar = b.item()
    if b_scalar is not None:
        if x.dtype in (torch.float16, torch.bfloat16):
            if b_scalar > _LARGE_B_THRESHOLD_FLOAT16:
                warnings.warn(
                    f"Large b value ({b_scalar}) with {x.dtype} may cause numerical "
                    f"overflow. Consider using float32 or float64 for better stability.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        elif x.dtype == torch.float32:
            if b_scalar > _LARGE_B_THRESHOLD_FLOAT32:
                warnings.warn(
                    f"Large b value ({b_scalar}) with float32 may cause numerical "
                    f"overflow. Consider using float64 for better stability.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    return torch.ops.torchscience.rosenbrock(x, a, b)
