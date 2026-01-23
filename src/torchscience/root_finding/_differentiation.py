"""Differentiation utilities for root finding."""

from typing import Callable

import torch
from torch import Tensor


def compute_derivative(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    df: Callable[[Tensor], Tensor] | None = None,
    method: str = "autodiff",
    h: float | None = None,
) -> Tensor:
    """Compute the derivative of a scalar function.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Scalar function to differentiate. For batched input (B,), returns (B,).
    x : Tensor
        Points at which to evaluate the derivative. Shape (B,) for batch size B.
    df : Callable[[Tensor], Tensor] or None
        Optional explicit derivative function. If provided, use it directly.
    method : str
        Differentiation method: "autodiff" or "finite_difference".
    h : float or None
        Step size for finite difference. Required if method="finite_difference".

    Returns
    -------
    Tensor
        Derivative values at x. Shape (B,).
    """
    # If explicit derivative provided, use it
    if df is not None:
        return df(x)

    if method == "autodiff":
        # Use torch.func.grad with vmap for batched computation
        def scalar_f(xi: Tensor) -> Tensor:
            """Evaluate f at a single scalar input."""
            return f(xi.unsqueeze(0)).squeeze(0)

        grad_f = torch.func.grad(scalar_f)
        return torch.vmap(grad_f)(x)

    elif method == "finite_difference":
        # Central difference: (f(x+h) - f(x-h)) / (2h)
        if h is None:
            h = 1e-7
        return (f(x + h) - f(x - h)) / (2 * h)

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'autodiff' or 'finite_difference'."
        )


def compute_second_derivative(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    ddf: Callable[[Tensor], Tensor] | None = None,
    method: str = "autodiff",
    h: float | None = None,
) -> Tensor:
    """Compute the second derivative of a scalar function.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Scalar function to differentiate twice. For batched input (B,), returns (B,).
    x : Tensor
        Points at which to evaluate the second derivative. Shape (B,).
    ddf : Callable[[Tensor], Tensor] or None
        Optional explicit second derivative function. If provided, use it directly.
    method : str
        Differentiation method: "autodiff" or "finite_difference".
    h : float or None
        Step size for finite difference. Required if method="finite_difference".

    Returns
    -------
    Tensor
        Second derivative values at x. Shape (B,).
    """
    # If explicit second derivative provided, use it
    if ddf is not None:
        return ddf(x)

    if method == "autodiff":
        # Apply grad twice using torch.func
        def scalar_f(xi: Tensor) -> Tensor:
            """Evaluate f at a single scalar input."""
            return f(xi.unsqueeze(0)).squeeze(0)

        grad_f = torch.func.grad(scalar_f)
        grad_grad_f = torch.func.grad(grad_f)
        return torch.vmap(grad_grad_f)(x)

    elif method == "finite_difference":
        # Central difference for second derivative: (f(x+h) - 2*f(x) + f(x-h)) / h^2
        if h is None:
            h = 1e-5
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'autodiff' or 'finite_difference'."
        )


def compute_jacobian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    jacobian: Callable[[Tensor], Tensor] | None = None,
    method: str = "autodiff",
) -> Tensor:
    """Compute the Jacobian matrix of a vector-valued function.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vector-valued function. For input (B, n), returns (B, m).
    x : Tensor
        Points at which to evaluate the Jacobian. Shape (B, n) or (n,).
    jacobian : Callable[[Tensor], Tensor] or None
        Optional explicit Jacobian function. If provided, use it directly.
        Should return shape (B, m, n) for input (B, n).
    method : str
        Differentiation method. Currently only "autodiff" is supported.

    Returns
    -------
    Tensor
        Jacobian matrices at x. Shape (B, m, n) where m is output dim, n is input dim.
        If input was unbatched (n,), output is (m, n).
    """
    # Handle unbatched input
    was_unbatched = x.dim() == 1
    if was_unbatched:
        x = x.unsqueeze(0)

    # If explicit Jacobian provided, use it
    if jacobian is not None:
        J = jacobian(x)
        if was_unbatched:
            J = J.squeeze(0)
        return J

    if method == "autodiff":
        # Compute Jacobian for each batch element using vmap
        def single_f(xi: Tensor) -> Tensor:
            """Evaluate f at a single (n,) input."""
            return f(xi.unsqueeze(0)).squeeze(0)

        J = torch.vmap(torch.func.jacrev(single_f))(x)
        if was_unbatched:
            J = J.squeeze(0)
        return J

    else:
        raise ValueError(f"Unknown method: {method}. Use 'autodiff'.")
