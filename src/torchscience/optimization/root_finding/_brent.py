from typing import Callable

import torch
from torch import Tensor


def _get_default_tol(dtype: torch.dtype) -> float:
    """Get dtype-aware default tolerance.

    Tolerances are chosen based on the number of significant digits
    available in each dtype:
    - bfloat16: ~3 digits (8-bit mantissa)
    - float16: ~3-4 digits (10-bit mantissa)
    - float32: ~7 digits (23-bit mantissa)
    - float64: ~16 digits (52-bit mantissa)
    """
    if dtype == torch.bfloat16:
        return 1e-2  # bfloat16 has less precision than float16
    elif dtype == torch.float16:
        return 1e-3
    elif dtype == torch.float32:
        return 1e-6
    else:  # float64
        return 1e-12


class _BrentImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through root-finding."""

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
            # This is done by computing the gradient of f w.r.t. its inputs
            # when evaluated at the root
            if fx.grad_fn is not None:
                # grad_output is dL/dx*, we need dL/dtheta
                # Using implicit function theorem: dx*/dtheta = -[df/dx]^{-1} * df/dtheta
                # So: dL/dtheta = dL/dx* * dx*/dtheta = -dL/dx* * [df/dx]^{-1} * df/dtheta
                #
                # We compute this by backpropagating through f with modified gradient
                # Safeguard against division by very small df/dx (near-horizontal tangent)
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
    result: Tensor, f: Callable[[Tensor], Tensor], orig_shape: tuple
) -> Tensor:
    """Attach implicit differentiation gradient if needed."""
    # Check if any parameter of f requires gradients
    try:
        test_input = result.detach().requires_grad_(True)
        with torch.enable_grad():
            test_output = f(test_input)
        needs_grad = test_output.requires_grad
    except Exception:
        needs_grad = False

    if not needs_grad:
        return result.reshape(orig_shape)

    # Make sure result has requires_grad=True for the autograd function
    # This is needed when result comes from endpoint detection (no iteration)
    if not result.requires_grad:
        result = result.clone().requires_grad_(True)

    result = _BrentImplicitGrad.apply(result, f, orig_shape)
    return result.reshape(orig_shape)


def brent(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    *,
    xtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> Tensor:
    """
    Find roots of f(x) = 0 using Brent's method.

    Brent's method combines bisection, secant, and inverse quadratic
    interpolation for robust and fast root-finding. It guarantees
    convergence like bisection but achieves superlinear convergence
    when possible.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape ``(N,)``, returns ``(N,)``.
        The function must be continuous on the interval [a, b].
    a, b : Tensor
        Bracket endpoints. Must have the same shape and satisfy
        ``f(a) * f(b) < 0`` for each element (opposite signs).
    xtol : float, optional
        Tolerance on interval width. Convergence requires ``|b - a| < xtol``.
        Default: dtype-aware (1e-3 for float16/bfloat16, 1e-6 for float32,
        1e-12 for float64).
    ftol : float, optional
        Tolerance on residual. Convergence requires ``|f(x)| < ftol``.
        Default: dtype-aware (same as xtol).
    maxiter : int, default=100
        Maximum iterations. Raises RuntimeError if exceeded.

    Returns
    -------
    Tensor
        Roots with the same shape as input ``a`` and ``b``.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different shapes, contain NaN/Inf,
        or if ``f(a)`` and ``f(b)`` have the same sign.
    RuntimeError
        If convergence is not achieved within maxiter iterations,
        or if the function returns NaN during iteration.

    Examples
    --------
    Find the square root of 2 (solve x^2 - 2 = 0):

    >>> import torch
    >>> from torchscience.optimization.root_finding import brent
    >>> f = lambda x: x**2 - 2
    >>> a, b = torch.tensor([1.0]), torch.tensor([2.0])
    >>> root = brent(f, a, b)
    >>> float(root)  # doctest: +ELLIPSIS
    1.414...

    Batched root-finding (find sqrt(2), sqrt(3), sqrt(4)):

    >>> c = torch.tensor([2.0, 3.0, 4.0])
    >>> f = lambda x: x**2 - c
    >>> a = torch.ones(3)
    >>> b = torch.full((3,), 10.0)
    >>> roots = brent(f, a, b)
    >>> [f"{v:.4f}" for v in roots.tolist()]
    ['1.4142', '1.7321', '2.0000']

    Find pi (solve sin(x) = 0 in [2, 4]):

    >>> f = lambda x: torch.sin(x)
    >>> a, b = torch.tensor([2.0]), torch.tensor([4.0])
    >>> float(brent(f, a, b))  # doctest: +ELLIPSIS
    3.141...

    Notes
    -----
    **Convergence Criterion**: Both ``xtol`` AND ``ftol`` must be satisfied
    for convergence. This ensures the root is both well-localized and
    the residual is small.

    **Autograd Support**: Gradients with respect to parameters in ``f``
    are computed via implicit differentiation using the implicit function
    theorem. If ``f(x*, theta) = 0``, then:

    .. math::

        \\frac{dx^*}{d\\theta} = -\\left[\\frac{\\partial f}{\\partial x}\\right]^{-1}
        \\frac{\\partial f}{\\partial \\theta}

    Example with autograd:

    >>> theta = torch.tensor([2.0], requires_grad=True)
    >>> f = lambda x: x**2 - theta  # root is sqrt(theta)
    >>> a, b = torch.tensor([0.0]), torch.tensor([3.0])
    >>> root = brent(f, a, b)
    >>> root.backward()
    >>> theta.grad  # d(sqrt(theta))/d(theta) = 1/(2*sqrt(theta))
    tensor([0.3536])

    **CUDA Support**: Works on any device (CPU or CUDA) as long as all
    inputs are on the same device.

    **Gradient Limitations**: Only first-order gradients have been validated.
    Second-order gradients (e.g., for Hessian computation) are not tested
    and may produce incorrect results.

    See Also
    --------
    scipy.optimize.brentq : SciPy's scalar Brent implementation
    """
    # Input validation
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have same shape, got {a.shape} and {b.shape}"
        )

    # Flatten for processing, remember original shape
    orig_shape = a.shape

    if a.numel() == 0:
        return _attach_implicit_grad(a.clone(), f, orig_shape)

    a = a.flatten()
    b = b.flatten()

    # Get tolerances
    dtype = a.dtype
    if xtol is None:
        xtol = _get_default_tol(dtype)
    if ftol is None:
        ftol = _get_default_tol(dtype)

    # Evaluate function at endpoints
    fa = f(a)
    fb = f(b)

    # Check for NaN/Inf in inputs
    if torch.any(~torch.isfinite(a)) or torch.any(~torch.isfinite(b)):
        raise ValueError("a and b must not contain NaN or Inf")

    # Check for NaN/Inf in function evaluations at endpoints
    if torch.any(~torch.isfinite(fa)) or torch.any(~torch.isfinite(fb)):
        raise ValueError("Function returned NaN or Inf at bracket endpoints")

    # Check for roots at endpoints first (before bracket validation)
    root = torch.where(fa == 0, a, torch.where(fb == 0, b, a))
    at_endpoint = (fa == 0) | (fb == 0)
    if torch.all(at_endpoint):
        return _attach_implicit_grad(root, f, orig_shape)

    # Check for valid brackets (only for non-endpoint cases)
    if torch.any(fa * fb >= 0):
        invalid = fa * fb >= 0
        invalid_indices = torch.where(invalid)[0].tolist()
        raise ValueError(
            f"Invalid bracket: f(a) and f(b) must have opposite signs. "
            f"{invalid.sum().item()} of {invalid.numel()} brackets are invalid "
            f"at indices {invalid_indices}."
        )

    # Ensure |f(a)| >= |f(b)| by swapping if needed
    swap_mask = torch.abs(fa) < torch.abs(fb)
    a, b = torch.where(swap_mask, b, a), torch.where(swap_mask, a, b)
    fa, fb = torch.where(swap_mask, fb, fa), torch.where(swap_mask, fa, fb)

    # Initialize state
    c = a.clone()
    fc = fa.clone()
    d = torch.zeros_like(a)
    mflag = torch.ones(a.shape, dtype=torch.bool)

    # Track which elements have converged
    converged = at_endpoint.clone()
    result = root.clone()

    for iteration in range(maxiter):
        # Check convergence: both xtol AND ftol must be satisfied
        interval_small = torch.abs(b - a) < xtol
        residual_small = torch.abs(fb) < ftol
        newly_converged = interval_small & residual_small & ~converged
        converged = converged | newly_converged
        result = torch.where(newly_converged, b, result)

        if torch.all(converged):
            return _attach_implicit_grad(result, f, orig_shape)

        # Only update unconverged elements
        active = ~converged

        # Compute s using inverse quadratic interpolation or secant method
        # Inverse quadratic interpolation
        use_iqp = (fa != fc) & (fb != fc) & active
        s_iqp = torch.where(
            use_iqp,
            (a * fb * fc) / ((fa - fb) * (fa - fc))
            + (b * fa * fc) / ((fb - fa) * (fb - fc))
            + (c * fa * fb) / ((fc - fa) * (fc - fb)),
            torch.zeros_like(a),
        )

        # Secant method
        use_secant = ~use_iqp & active
        s_secant = torch.where(
            use_secant & (fb != fa),
            b - fb * (b - a) / (fb - fa),
            torch.zeros_like(a),
        )

        s = torch.where(use_iqp, s_iqp, s_secant)

        # Brent's method uses 5 conditions to decide when to fall back to bisection.
        # These conditions ensure that the interpolation step makes sufficient
        # progress; otherwise, bisection provides guaranteed convergence.

        # Condition 1: s must lie in the interval ((3a+b)/4, b).
        # This ensures the new estimate is within a reasonable range and not
        # outside the bracket or too close to endpoint a.
        min_range = torch.minimum((3 * a + b) / 4, b)
        max_range = torch.maximum((3 * a + b) / 4, b)
        cond1 = ~((min_range < s) & (s < max_range))

        # Condition 2: If bisection was used in the previous step (mflag=True),
        # and the new step |s - b| is at least half of |b - c|, then bisection
        # would have made at least as much progress. Use bisection instead.
        cond2 = mflag & (torch.abs(s - b) >= torch.abs(b - c) / 2)

        # Condition 3: If interpolation was used previously (mflag=False),
        # and the new step |s - b| is at least half of |c - d|, the interpolation
        # is not converging fast enough. Fall back to bisection.
        cond3 = ~mflag & (torch.abs(s - b) >= torch.abs(c - d) / 2)

        # Condition 4: If bisection was used previously and the interval |b - c|
        # is already smaller than xtol, further bisection won't help.
        # This prevents infinite loops when the interval is very small.
        cond4 = mflag & (torch.abs(b - c) < xtol)

        # Condition 5: If interpolation was used previously and |c - d| < xtol,
        # the previous interpolation step was too small to be useful.
        # Fall back to bisection for guaranteed progress.
        cond5 = ~mflag & (torch.abs(c - d) < xtol)

        use_bisection = (cond1 | cond2 | cond3 | cond4 | cond5) & active

        # Bisection
        s = torch.where(use_bisection, (a + b) / 2, s)
        mflag = torch.where(active, use_bisection, mflag)

        # Evaluate function at s
        fs = f(s)

        # Check for NaN in function evaluation
        if torch.any(torch.isnan(fs) & active):
            raise RuntimeError("Function returned NaN during iteration")

        # d is now the value of c from the previous iteration
        d = torch.where(active, c, d)

        # c takes the value of b
        c = torch.where(active, b, c)
        fc = torch.where(active, fb, fc)

        # Update the bracket
        update_b = (fa * fs < 0) & active
        update_a = ~update_b & active

        # Update b
        b = torch.where(update_b, s, b)
        fb = torch.where(update_b, fs, fb)

        # Update a
        a = torch.where(update_a, s, a)
        fa = torch.where(update_a, fs, fa)

        # Ensure |f(a)| >= |f(b)| by swapping if needed
        swap = (torch.abs(fa) < torch.abs(fb)) & active
        a_new = torch.where(swap, b, a)
        b_new = torch.where(swap, a, b)
        fa_new = torch.where(swap, fb, fa)
        fb_new = torch.where(swap, fa, fb)

        a = a_new
        b = b_new
        fa = fa_new
        fb = fb_new

    raise RuntimeError(f"brent: failed to converge in {maxiter} iterations")
