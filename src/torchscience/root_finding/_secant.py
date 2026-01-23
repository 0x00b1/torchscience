"""Secant root finding method."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import check_convergence, default_tolerances


class _SecantImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through Secant root-finding."""

    @staticmethod
    def forward(ctx, root: Tensor, f_callable, orig_shape) -> Tensor:
        ctx.f_callable = f_callable
        ctx.orig_shape = orig_shape
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, None]:
        (root,) = ctx.saved_tensors

        # Compute df/dx at the root
        x = root.detach().requires_grad_(True)
        with torch.enable_grad():
            fx = ctx.f_callable(x)
            # Compute df/dx
            df_dx = torch.autograd.grad(
                fx,
                x,
                grad_outputs=torch.ones_like(fx),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Compute df/dtheta (gradient w.r.t. parameters)
            # Using implicit function theorem: dx*/dtheta = -[df/dx]^{-1} * df/dtheta
            # So: dL/dtheta = dL/dx* * dx*/dtheta = -dL/dx* * [df/dx]^{-1} * df/dtheta
            if fx.grad_fn is not None:
                # Safeguard against division by very small df/dx
                eps = torch.finfo(df_dx.dtype).eps * 10
                safe_df_dx = torch.where(
                    torch.abs(df_dx) < eps,
                    torch.sign(df_dx) * eps,
                    df_dx,
                )
                # Handle case where df_dx is exactly zero (sign returns 0)
                safe_df_dx = torch.where(safe_df_dx == 0, eps, safe_df_dx)
                modified_grad = -grad_output / safe_df_dx
                torch.autograd.backward(fx, modified_grad)

        return None, None, None


def _attach_implicit_grad(
    result: Tensor,
    converged: Tensor,
    f: Callable[[Tensor], Tensor],
    orig_shape: tuple,
) -> tuple[Tensor, Tensor]:
    """Attach implicit differentiation gradient if needed.

    Returns
    -------
    tuple[Tensor, Tensor]
        (result, converged) both reshaped to orig_shape.
    """
    # Check if any parameter of f requires gradients
    try:
        test_input = result.detach().requires_grad_(True)
        with torch.enable_grad():
            test_output = f(test_input)
        needs_grad = test_output.requires_grad
    except Exception:
        needs_grad = False

    if not needs_grad:
        return result.reshape(orig_shape), converged.reshape(orig_shape)

    # Make sure result has requires_grad=True for the autograd function
    if not result.requires_grad:
        result = result.clone().requires_grad_(True)

    result = _SecantImplicitGrad.apply(result, f, orig_shape)
    return result.reshape(orig_shape), converged.reshape(orig_shape)


def secant(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    x1: Tensor | None = None,
    *,
    xtol: float | None = None,
    rtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 50,
) -> tuple[Tensor, Tensor]:
    """
    Find roots of f(x) = 0 using the Secant method.

    The Secant method approximates the derivative using finite differences:
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))

    Unlike Newton's method, this does not require computing derivatives,
    making it useful when derivatives are expensive or unavailable.
    The method has superlinear convergence of order approximately 1.618
    (the golden ratio).

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape ``(N,)``, returns ``(N,)``.
    x0 : Tensor
        First initial guess for the root. Shape ``(N,)`` or any shape that will
        be flattened for processing.
    x1 : Tensor, optional
        Second initial guess. If None (default), uses ``x0 * 1.0001 + 1e-4``
        as a small perturbation from x0.
    xtol : float, optional
        Absolute tolerance on x change. Convergence requires
        ``|x_new - x_old| < xtol + rtol * |x_old|``.
        Default: dtype-aware (1e-3 for float16/bfloat16, 1e-6 for float32,
        1e-12 for float64).
    rtol : float, optional
        Relative tolerance on x change. Combined with xtol for convergence.
        Default: dtype-aware (1e-2 for float16/bfloat16, 1e-5 for float32,
        1e-9 for float64).
    ftol : float, optional
        Tolerance on residual. Convergence requires ``|f(x)| < ftol``.
        Default: dtype-aware (same as xtol).
    maxiter : int, default=50
        Maximum iterations. Non-converged elements will have converged=False.

    Returns
    -------
    tuple[Tensor, Tensor]
        - **root** -- Roots with the same shape as input ``x0``.
          For non-converged elements, this is the best estimate.
        - **converged** -- Boolean tensor with the same shape indicating
          which elements converged within maxiter iterations.

    Examples
    --------
    Find the square root of 2 (solve x^2 - 2 = 0):

    >>> import torch
    >>> from torchscience.root_finding import secant
    >>> f = lambda x: x**2 - 2
    >>> x0 = torch.tensor([1.5])
    >>> root, converged = secant(f, x0)
    >>> float(root)  # doctest: +ELLIPSIS
    1.414...
    >>> converged.all()
    tensor(True)

    Batched root-finding (find sqrt(2), sqrt(3), sqrt(4)):

    >>> c = torch.tensor([2.0, 3.0, 4.0])
    >>> f = lambda x: x**2 - c
    >>> x0 = torch.tensor([1.5, 1.5, 1.5])
    >>> roots, converged = secant(f, x0)
    >>> [f"{v:.4f}" for v in roots.tolist()]
    ['1.4142', '1.7321', '2.0000']

    Notes
    -----
    **Convergence**: The Secant method has superlinear convergence with order
    approximately 1.618 (the golden ratio), which is slower than Newton's
    quadratic convergence but does not require derivative computation.

    **Zero Denominator Safeguard**: When ``|f(x_n) - f(x_{n-1})| < eps``,
    the method uses ``sign(diff) * eps`` to prevent division by zero.

    **Automatic x1 Generation**: When x1 is not provided, a small perturbation
    ``x0 * 1.0001 + 1e-4`` is used. This ensures the two points are distinct
    even when x0 is zero.

    **Autograd Support**: Gradients with respect to parameters in ``f``
    are computed via implicit differentiation using the implicit function
    theorem.

    **CUDA Support**: Works on any device (CPU or CUDA) as long as all
    inputs are on the same device.

    See Also
    --------
    newton : Newton-Raphson method (requires derivatives, quadratic convergence)
    brent : Brent's method (bracketed, guaranteed convergence)
    """
    # Flatten for processing, remember original shape
    orig_shape = x0.shape

    if x0.numel() == 0:
        empty_converged = torch.ones(
            x0.shape, dtype=torch.bool, device=x0.device
        )
        return _attach_implicit_grad(
            x0.clone(), empty_converged, f, orig_shape
        )

    x_prev = x0.flatten().clone()

    # Generate x1 if not provided
    if x1 is None:
        x_curr = x_prev * 1.0001 + 1e-4
    else:
        x_curr = x1.flatten().clone()

    # Get tolerances
    dtype = x_prev.dtype
    defaults = default_tolerances(dtype)
    if xtol is None:
        xtol = defaults["xtol"]
    if rtol is None:
        rtol = defaults["rtol"]
    if ftol is None:
        ftol = defaults["ftol"]

    # Track which elements have converged
    converged = torch.zeros(
        x_prev.shape, dtype=torch.bool, device=x_prev.device
    )
    result = x_curr.clone()

    # Machine epsilon for safeguarding zero denominator
    eps = torch.finfo(dtype).eps * 10

    # Evaluate function at both points
    f_prev = f(x_prev)
    f_curr = f(x_curr)

    for _ in range(maxiter):
        # Compute denominator: f(x_n) - f(x_{n-1})
        denom = f_curr - f_prev

        # Safeguard against zero denominator
        safe_denom = torch.where(
            torch.abs(denom) < eps,
            torch.sign(denom) * eps,
            denom,
        )
        # Handle case where denom is exactly zero (sign returns 0)
        safe_denom = torch.where(safe_denom == 0, eps, safe_denom)

        # Secant step: x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        x_new = x_curr - f_curr * (x_curr - x_prev) / safe_denom

        # Evaluate function at new point
        f_new = f(x_new)

        # Check convergence
        newly_converged = check_convergence(
            x_curr, x_new, f_new, xtol, rtol, ftol
        )
        newly_converged = newly_converged & ~converged

        # Update results for newly converged elements
        result = torch.where(newly_converged, x_new, result)
        converged = converged | newly_converged

        if torch.all(converged):
            return _attach_implicit_grad(result, converged, f, orig_shape)

        # Update for next iteration (only unconverged elements)
        # Shift: x_{n-1} <- x_n, x_n <- x_{n+1}
        x_prev = torch.where(converged, x_prev, x_curr)
        f_prev = torch.where(converged, f_prev, f_curr)
        x_curr = torch.where(converged, x_curr, x_new)
        f_curr = torch.where(converged, f_curr, f_new)

    # Return best estimate for non-converged elements
    result = torch.where(converged, result, x_curr)
    return _attach_implicit_grad(result, converged, f, orig_shape)
