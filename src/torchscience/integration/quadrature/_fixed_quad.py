"""Fixed-order Gaussian quadrature."""

from typing import Callable, Union

import torch
from torch import Tensor

from torchscience.integration.quadrature._rules import GaussLegendre


class _FixedQuadFunction(torch.autograd.Function):
    """
    Custom autograd function for fixed quadrature with Leibniz rule gradients.

    Implements the Leibniz integral rule for differentiating through integration limits:
        d/db integral_a^b f(x) dx = f(b)
        d/da integral_a^b f(x) dx = -f(a)
    """

    @staticmethod
    def forward(
        ctx, a: Tensor, b: Tensor, f: Callable, rule: GaussLegendre
    ) -> Tensor:
        # Compute integral
        result = rule.integrate(f, a, b)

        # Save for backward
        ctx.save_for_backward(a, b)
        ctx.f = f

        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        a, b = ctx.saved_tensors
        f = ctx.f

        grad_a = None
        grad_b = None

        # Leibniz integral rule:
        # d/db integral_a^b f(x) dx = f(b)
        # d/da integral_a^b f(x) dx = -f(a)

        if ctx.needs_input_grad[0]:
            grad_a = -grad_output * f(a)

        if ctx.needs_input_grad[1]:
            grad_b = grad_output * f(b)

        return grad_a, grad_b, None, None


def fixed_quad(
    f: Callable[[Tensor], Tensor],
    a: Union[float, Tensor],
    b: Union[float, Tensor],
    *,
    n: int = 5,
) -> Tensor:
    """
    Compute definite integral using fixed-order Gaussian quadrature.

    Parameters
    ----------
    f : callable
        Integrand function. Receives tensor of quadrature points,
        returns tensor of function values.
    a, b : float or Tensor
        Lower and upper integration bounds. Can be batched.
    n : int
        Number of quadrature points.

    Returns
    -------
    Tensor
        Integral approximation. Shape matches broadcast(a, b).

    Notes
    -----
    Differentiable with respect to:
    - Integration limits a, b (via Leibniz rule)
    - Parameters captured in f's closure

    The Leibniz integral rule states:
        d/db integral_a^b f(x) dx = f(b)
        d/da integral_a^b f(x) dx = -f(a)

    For gradients through parameters in f, the gradient flows automatically
    through the function evaluation at quadrature points.

    Gauss-Legendre quadrature with n points is exact for polynomials
    of degree <= 2n-1.

    Examples
    --------
    >>> # Basic integration
    >>> fixed_quad(torch.sin, 0, torch.pi)  # approximately 2.0

    >>> # Batched limits
    >>> b = torch.linspace(1, 10, 100)
    >>> fixed_quad(torch.sin, 0, b)  # Shape: (100,)

    >>> # Differentiable through parameters
    >>> theta = torch.tensor(2.0, requires_grad=True)
    >>> result = fixed_quad(lambda x: theta * x**2, 0, 1)
    >>> result.backward()
    >>> theta.grad  # 1/3
    """
    rule = GaussLegendre(n)

    # Infer dtype and device
    if isinstance(a, Tensor):
        dtype = a.dtype
        device = a.device
    elif isinstance(b, Tensor):
        dtype = b.dtype
        device = b.device
    else:
        dtype = torch.float64
        device = torch.device("cpu")

    # Ensure tensors
    if not isinstance(a, Tensor):
        a = torch.tensor(a, dtype=dtype, device=device)
    if not isinstance(b, Tensor):
        b = torch.tensor(b, dtype=dtype, device=device)

    # Check if we need custom gradient handling for limits
    if a.requires_grad or b.requires_grad:
        return _FixedQuadFunction.apply(a, b, f, rule)
    else:
        # No need for custom backward, just use regular autograd
        return rule.integrate(f, a, b)
