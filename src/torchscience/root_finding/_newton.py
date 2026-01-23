"""Newton-Raphson root finding method."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import check_convergence, default_tolerances


def _compute_derivative_batched(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    df: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Compute the derivative of a batched scalar function.

    This function handles batched functions properly, where f captures external
    batched parameters (e.g., f = lambda x: x**2 - c where c is batched).

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Scalar function that maps (B,) -> (B,).
    x : Tensor
        Points at which to evaluate the derivative. Shape (B,).
    df : Callable[[Tensor], Tensor] or None
        Optional explicit derivative function.

    Returns
    -------
    Tensor
        Derivative values at x. Shape (B,).
    """
    if df is not None:
        return df(x)

    # Use torch.autograd.grad to compute gradient of sum(f(x)) w.r.t. x
    # This gives us df/dx for each element since f is element-wise
    x_grad = x.detach().requires_grad_(True)
    with torch.enable_grad():
        fx = f(x_grad)
        # Compute gradient of sum(f(x)) w.r.t. x
        # For element-wise f, this gives [df/dx_1, df/dx_2, ...]
        grad = torch.autograd.grad(
            fx.sum(),
            x_grad,
            create_graph=False,
            retain_graph=False,
        )[0]
    return grad


class _NewtonImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through Newton root-finding."""

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

    result = _NewtonImplicitGrad.apply(result, f, orig_shape)
    return result.reshape(orig_shape), converged.reshape(orig_shape)


def newton(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    df: Callable[[Tensor], Tensor] | None = None,
    xtol: float | None = None,
    rtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 50,
) -> tuple[Tensor, Tensor]:
    """
    Find roots of f(x) = 0 using Newton-Raphson method.

    Newton's method uses the iteration x_{n+1} = x_n - f(x_n) / f'(x_n)
    to find roots. It converges quadratically when starting near a root,
    but may diverge if the initial guess is far from any root or if the
    derivative is zero near the root.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape ``(N,)``, returns ``(N,)``.
        The function should be differentiable for autodiff to work.
    x0 : Tensor
        Initial guess for the root. Shape ``(N,)`` or any shape that will
        be flattened for processing.
    df : Callable[[Tensor], Tensor], optional
        Explicit derivative function. If None (default), the derivative is
        computed using autodiff. Providing an explicit derivative can be
        faster for functions where autodiff is expensive.
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
    >>> from torchscience.root_finding import newton
    >>> f = lambda x: x**2 - 2
    >>> x0 = torch.tensor([1.5])
    >>> root, converged = newton(f, x0)
    >>> float(root)  # doctest: +ELLIPSIS
    1.414...
    >>> converged.all()
    tensor(True)

    Using explicit derivative (avoids autodiff overhead):

    >>> f = lambda x: x**2 - 2
    >>> df = lambda x: 2 * x
    >>> x0 = torch.tensor([1.5])
    >>> root, converged = newton(f, x0, df=df)
    >>> float(root)  # doctest: +ELLIPSIS
    1.414...

    Batched root-finding (find sqrt(2), sqrt(3), sqrt(4)):

    >>> c = torch.tensor([2.0, 3.0, 4.0])
    >>> f = lambda x: x**2 - c
    >>> x0 = torch.tensor([1.5, 1.5, 1.5])
    >>> roots, converged = newton(f, x0)
    >>> [f"{v:.4f}" for v in roots.tolist()]
    ['1.4142', '1.7321', '2.0000']

    Notes
    -----
    **Convergence**: Newton's method has quadratic convergence near simple
    roots, meaning the number of correct digits roughly doubles each iteration.
    However, it can fail to converge or converge slowly when:

    - The initial guess is far from the root
    - The derivative is zero or very small near the root
    - The function has multiple roots and the method oscillates

    **Zero Derivative Safeguard**: When ``|f'(x)| < eps``, the method uses
    ``sign(f'(x)) * eps`` to prevent division by zero. This allows the
    iteration to continue but may result in large steps.

    **Autograd Support**: Gradients with respect to parameters in ``f``
    are computed via implicit differentiation using the implicit function
    theorem. If ``f(x*, theta) = 0``, then:

    .. math::

        \\frac{dx^*}{d\\theta} = -\\left[\\frac{\\partial f}{\\partial x}\\right]^{-1}
        \\frac{\\partial f}{\\partial \\theta}

    **CUDA Support**: Works on any device (CPU or CUDA) as long as all
    inputs are on the same device.

    See Also
    --------
    brent : Brent's method (bracketed, guaranteed convergence)
    scipy.optimize.newton : SciPy's Newton implementation
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

    x = x0.flatten().clone()

    # Get tolerances
    dtype = x.dtype
    defaults = default_tolerances(dtype)
    if xtol is None:
        xtol = defaults["xtol"]
    if rtol is None:
        rtol = defaults["rtol"]
    if ftol is None:
        ftol = defaults["ftol"]

    # Track which elements have converged
    converged = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    result = x.clone()

    # Machine epsilon for safeguarding zero derivative
    eps = torch.finfo(dtype).eps * 10

    for _ in range(maxiter):
        # Evaluate function
        fx = f(x)

        # Compute derivative using autodiff or explicit df
        dfx = _compute_derivative_batched(f, x, df=df)

        # Safeguard against zero derivative
        safe_dfx = torch.where(
            torch.abs(dfx) < eps,
            torch.sign(dfx) * eps,
            dfx,
        )
        # Handle case where dfx is exactly zero (sign returns 0)
        safe_dfx = torch.where(safe_dfx == 0, eps, safe_dfx)

        # Newton step: x_new = x - f(x) / f'(x)
        x_new = x - fx / safe_dfx

        # Check convergence
        newly_converged = check_convergence(
            x, x_new, f(x_new), xtol, rtol, ftol
        )
        newly_converged = newly_converged & ~converged

        # Update results for newly converged elements
        result = torch.where(newly_converged, x_new, result)
        converged = converged | newly_converged

        if torch.all(converged):
            return _attach_implicit_grad(result, converged, f, orig_shape)

        # Update x for next iteration (only unconverged elements)
        x = torch.where(converged, x, x_new)

    # Return best estimate for non-converged elements
    result = torch.where(converged, result, x)
    return _attach_implicit_grad(result, converged, f, orig_shape)
