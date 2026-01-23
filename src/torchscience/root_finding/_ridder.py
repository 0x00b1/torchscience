"""Ridder's bracketed root-finding method."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import default_tolerances


class _RidderImplicitGrad(torch.autograd.Function):
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
                torch.autograd.backward(fx, modified_grad, create_graph=True)

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

    result = _RidderImplicitGrad.apply(result, f, orig_shape)
    return result.reshape(orig_shape), converged.reshape(orig_shape)


def ridder(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    *,
    xtol: float | None = None,
    rtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> tuple[Tensor, Tensor]:
    """
    Find roots of f(x) = 0 using Ridder's method.

    Ridder's method is a bracketed root-finding algorithm that achieves
    quadratic convergence by using an exponential interpolation. It is
    more robust than the secant method while achieving faster convergence
    than bisection.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vectorized function. Takes tensor of shape ``(N,)``, returns ``(N,)``.
        The function must be continuous on the interval [a, b].
    a, b : Tensor
        Bracket endpoints. Must have the same shape and satisfy
        ``f(a) * f(b) < 0`` for each element (opposite signs).
    xtol : float, optional
        Absolute tolerance on interval width. Convergence requires
        ``|b - a| < xtol + rtol * |b|``.
        Default: dtype-aware (1e-3 for float16/bfloat16, 1e-6 for float32,
        1e-12 for float64).
    rtol : float, optional
        Relative tolerance on interval width. Combined with xtol for
        convergence checking.
        Default: dtype-aware (1e-2 for float16/bfloat16, 1e-5 for float32,
        1e-9 for float64).
    ftol : float, optional
        Tolerance on residual. Convergence requires ``|f(x)| < ftol``.
        Default: dtype-aware (same as xtol).
    maxiter : int, default=100
        Maximum iterations. Non-converged elements will have converged=False.

    Returns
    -------
    tuple[Tensor, Tensor]
        - **root** -- Roots with the same shape as input ``a`` and ``b``.
          For non-converged elements, this is the best estimate.
        - **converged** -- Boolean tensor with the same shape indicating
          which elements converged within maxiter iterations.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different shapes, contain NaN/Inf,
        or if ``f(a)`` and ``f(b)`` have the same sign.
    RuntimeError
        If the function returns NaN during iteration.

    Examples
    --------
    Find the square root of 2 (solve x^2 - 2 = 0):

    >>> import torch
    >>> from torchscience.root_finding import ridder
    >>> f = lambda x: x**2 - 2
    >>> a, b = torch.tensor([1.0]), torch.tensor([2.0])
    >>> root, converged = ridder(f, a, b)
    >>> float(root)  # doctest: +ELLIPSIS
    1.414...
    >>> converged.all()
    tensor(True)

    Batched root-finding (find sqrt(2), sqrt(3), sqrt(4)):

    >>> c = torch.tensor([2.0, 3.0, 4.0])
    >>> f = lambda x: x**2 - c
    >>> a = torch.ones(3)
    >>> b = torch.full((3,), 10.0)
    >>> roots, converged = ridder(f, a, b)
    >>> [f"{v:.4f}" for v in roots.tolist()]
    ['1.4142', '1.7321', '2.0000']

    Find pi (solve sin(x) = 0 in [2, 4]):

    >>> f = lambda x: torch.sin(x)
    >>> a, b = torch.tensor([2.0]), torch.tensor([4.0])
    >>> root, _ = ridder(f, a, b)
    >>> float(root)  # doctest: +ELLIPSIS
    3.141...

    Notes
    -----
    **Algorithm**: Ridder's method works by computing a midpoint m = (a + b) / 2
    and then finding an improved estimate using exponential interpolation:

    .. math::

        x_{new} = m + \\text{sign}(f(a) - f(b)) \\cdot \\frac{(m - a) \\cdot f(m)}{s}

    where :math:`s = \\sqrt{f(m)^2 - f(a) \\cdot f(b)}`. This formula guarantees
    that the new point lies within the bracket [a, b].

    **Convergence Criterion**: Both the interval width (xtol + rtol * |b|)
    AND ``ftol`` must be satisfied for convergence. This ensures the root
    is both well-localized and the residual is small.

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
    >>> root, _ = ridder(f, a, b)
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
    scipy.optimize.ridder : SciPy's scalar Ridder implementation
    brent : Brent's method (typically faster for smooth functions)
    """
    # Input validation
    if a.shape != b.shape:
        raise ValueError(
            f"a and b must have same shape, got {a.shape} and {b.shape}"
        )

    # Flatten for processing, remember original shape
    orig_shape = a.shape

    if a.numel() == 0:
        empty_converged = torch.ones(
            a.shape, dtype=torch.bool, device=a.device
        )
        return _attach_implicit_grad(a.clone(), empty_converged, f, orig_shape)

    a = a.flatten()
    b = b.flatten()

    # Get tolerances
    dtype = a.dtype
    defaults = default_tolerances(dtype)
    if xtol is None:
        xtol = defaults["xtol"]
    if rtol is None:
        rtol = defaults["rtol"]
    if ftol is None:
        ftol = defaults["ftol"]

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
        endpoint_converged = torch.ones(
            a.shape, dtype=torch.bool, device=a.device
        )
        return _attach_implicit_grad(root, endpoint_converged, f, orig_shape)

    # Check for valid brackets (only for non-endpoint cases)
    if torch.any(fa * fb >= 0):
        invalid = fa * fb >= 0
        invalid_indices = torch.where(invalid)[0].tolist()
        raise ValueError(
            f"Invalid bracket: f(a) and f(b) must have opposite signs. "
            f"{invalid.sum().item()} of {invalid.numel()} brackets are invalid "
            f"at indices {invalid_indices}."
        )

    # Track which elements have converged
    converged = at_endpoint.clone()
    result = root.clone()

    # Track the best x_new for each element (for returning)
    best_x = (a + b) / 2

    for iteration in range(maxiter):
        # Check convergence: both interval tolerance AND ftol must be satisfied
        interval_small = torch.abs(b - a) < xtol + rtol * torch.abs(b)
        f_best = f(best_x)
        residual_small = torch.abs(f_best) < ftol
        newly_converged = interval_small & residual_small & ~converged
        converged = converged | newly_converged
        result = torch.where(newly_converged, best_x, result)

        if torch.all(converged):
            return _attach_implicit_grad(result, converged, f, orig_shape)

        # Only update unconverged elements
        active = ~converged

        # Ridder's method:
        # 1. Compute midpoint m = (a + b) / 2
        m = (a + b) / 2
        fm = f(m)

        # Check for NaN in function evaluation
        if torch.any(torch.isnan(fm) & active):
            raise RuntimeError("Function returned NaN during iteration")

        # 2. Compute s = sqrt(f(m)^2 - f(a)*f(b))
        # This discriminant is always non-negative when the bracket is valid
        discriminant = fm**2 - fa * fb
        # Clamp to avoid numerical issues with sqrt of small negative numbers
        discriminant = torch.clamp(discriminant, min=0.0)
        s = torch.sqrt(discriminant)

        # 3. Compute x_new = m + sign(f(a) - f(b)) * (m - a) * f(m) / s
        # Handle case where s is zero (degenerate case)
        sign_term = torch.sign(fa - fb)
        # When s is zero, fall back to midpoint
        s_safe = torch.where(s == 0, torch.ones_like(s), s)
        x_new = torch.where(
            (s == 0) | ~active,
            m,
            m + sign_term * (m - a) * fm / s_safe,
        )

        # Clamp x_new to bracket [a, b] to ensure we stay within bounds
        x_new = torch.clamp(x_new, torch.minimum(a, b), torch.maximum(a, b))

        # Update best_x for active elements
        best_x = torch.where(active, x_new, best_x)

        # Evaluate function at new point
        f_new = f(x_new)

        # Check for NaN in function evaluation
        if torch.any(torch.isnan(f_new) & active):
            raise RuntimeError("Function returned NaN during iteration")

        # Check if we found an exact root (or within ftol)
        found_root = (torch.abs(f_new) < ftol) & active
        newly_converged = found_root & ~converged
        converged = converged | newly_converged
        result = torch.where(newly_converged, x_new, result)
        active = ~converged

        if torch.all(converged):
            return _attach_implicit_grad(result, converged, f, orig_shape)

        # 4. Update bracket based on sign of f(x_new)
        # The new bracket should contain the root and have opposite signs at endpoints
        #
        # We need to pick two points from {a, m, x_new, b} such that f has opposite signs
        # Ridder's method produces x_new between a and b, and we need to form the
        # tightest bracket around the root.

        # Check if f(m) and f(x_new) have opposite signs
        m_xnew_bracket = fm * f_new < 0

        # Check if f(a) and f(x_new) have opposite signs
        a_xnew_bracket = fa * f_new < 0

        # Case 1: Use [m, x_new] or [x_new, m] as new bracket
        # Case 2: Use [a, x_new] as new bracket
        # Case 3: Use [x_new, b] as new bracket

        # For case 1: bracket is [min(m, x_new), max(m, x_new)]
        case1 = m_xnew_bracket & active
        # For case 2: bracket is [a, x_new] (f(a) and f(x_new) have opposite signs)
        case2 = a_xnew_bracket & ~case1 & active
        # For case 3: bracket is [x_new, b] (f(x_new) and f(b) have opposite signs)
        case3 = ~case1 & ~case2 & active

        # Initialize new values to current values
        a_new = a.clone()
        fa_new = fa.clone()
        b_new = b.clone()
        fb_new = fb.clone()

        # Case 1: bracket is [min(m, x_new), max(m, x_new)]
        left_is_m = m < x_new
        a_new = torch.where(case1 & left_is_m, m, a_new)
        fa_new = torch.where(case1 & left_is_m, fm, fa_new)
        b_new = torch.where(case1 & left_is_m, x_new, b_new)
        fb_new = torch.where(case1 & left_is_m, f_new, fb_new)

        a_new = torch.where(case1 & ~left_is_m, x_new, a_new)
        fa_new = torch.where(case1 & ~left_is_m, f_new, fa_new)
        b_new = torch.where(case1 & ~left_is_m, m, b_new)
        fb_new = torch.where(case1 & ~left_is_m, fm, fb_new)

        # Case 2: bracket is [a, x_new]
        # a stays the same, b becomes x_new
        b_new = torch.where(case2, x_new, b_new)
        fb_new = torch.where(case2, f_new, fb_new)

        # Case 3: bracket is [x_new, b]
        # a becomes x_new, b stays the same
        a_new = torch.where(case3, x_new, a_new)
        fa_new = torch.where(case3, f_new, fa_new)

        # Apply updates
        a = a_new
        fa = fa_new
        b = b_new
        fb = fb_new

    # Final convergence check after all iterations
    interval_small = torch.abs(b - a) < xtol + rtol * torch.abs(b)
    f_best = f(best_x)
    residual_small = torch.abs(f_best) < ftol
    newly_converged = interval_small & residual_small & ~converged
    converged = converged | newly_converged
    result = torch.where(newly_converged, best_x, result)

    # Return best estimate for non-converged elements
    result = torch.where(converged, result, best_x)
    return _attach_implicit_grad(result, converged, f, orig_shape)
