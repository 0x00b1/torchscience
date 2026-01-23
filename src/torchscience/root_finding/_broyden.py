"""Broyden's quasi-Newton method for systems of equations."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import default_tolerances


class _BroydenImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through Broyden root-finding.

    For systems of equations, the implicit function theorem gives:
        dx*/dtheta = -J^{-1} @ df/dtheta
    where J = df/dx is the Jacobian matrix at the root.
    """

    @staticmethod
    def forward(ctx, root: Tensor, f_callable, was_unbatched: bool) -> Tensor:
        ctx.f_callable = f_callable
        ctx.was_unbatched = was_unbatched
        ctx.save_for_backward(root)
        return root

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, None, None]:
        (root,) = ctx.saved_tensors

        # root already comes in batched form (B, n) from the main function
        # No need to add batch dimension here
        root_batched = root
        grad_batched = grad_output

        batch_size = root_batched.shape[0]
        n = root_batched.shape[1]

        # Compute Jacobian df/dx at the root
        x = root_batched.detach().requires_grad_(True)
        with torch.enable_grad():
            fx = ctx.f_callable(x)

            # Compute full Jacobian matrix for each batch element
            # J[b, i, j] = d(f_i)/d(x_j) for batch b
            jacobian = torch.zeros(
                batch_size, n, n, dtype=x.dtype, device=x.device
            )
            for i in range(n):
                # Compute gradient of f_i w.r.t. x
                grad_fi = torch.autograd.grad(
                    fx[..., i].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                jacobian[:, i, :] = grad_fi

            # Apply implicit function theorem:
            # dL/dtheta = dL/dx* @ dx*/dtheta = -dL/dx* @ J^{-1} @ df/dtheta
            # We compute: v = -J^{-T} @ grad_output, then backprop v through f
            # This is because: dL/dtheta = grad_output^T @ (-J^{-1} @ df/dtheta)
            #                            = (-J^{-T} @ grad_output)^T @ df/dtheta
            try:
                # Solve J^T @ v = -grad_output for v
                # Equivalent to v = -J^{-T} @ grad_output
                neg_grad = -grad_batched.unsqueeze(-1)  # (B, n, 1)
                v = torch.linalg.solve(
                    jacobian.transpose(-2, -1), neg_grad
                ).squeeze(-1)  # (B, n)
            except RuntimeError:
                # Fallback to pseudoinverse for singular Jacobian
                J_pinv_T = torch.linalg.pinv(jacobian).transpose(-2, -1)
                v = torch.einsum("bij,bj->bi", -J_pinv_T, grad_batched)

            # Backpropagate v through f to get df/dtheta contribution
            if fx.grad_fn is not None:
                # Sum over batch and system dimensions with v as weights
                torch.autograd.backward(fx, v, create_graph=True)

        return None, None, None


def _attach_implicit_grad(
    result: Tensor,
    converged: Tensor,
    f: Callable[[Tensor], Tensor],
    was_unbatched: bool,
) -> tuple[Tensor, Tensor]:
    """Attach implicit differentiation gradient if needed.

    Parameters
    ----------
    result : Tensor
        The root found by Broyden's method. Shape (B, n) or (n,).
    converged : Tensor
        Boolean tensor indicating convergence. Shape (B,) or scalar.
    f : Callable
        The function whose root was found.
    was_unbatched : bool
        Whether the original input was unbatched (1D).

    Returns
    -------
    tuple[Tensor, Tensor]
        (result, converged) with implicit grad attached if needed.
    """
    # Check if any parameter of f requires gradients
    try:
        # Ensure we have batch dimension for the test
        if was_unbatched:
            test_input = result.unsqueeze(0).detach().requires_grad_(True)
        else:
            test_input = result.detach().requires_grad_(True)
        with torch.enable_grad():
            test_output = f(test_input)
        needs_grad = test_output.requires_grad
    except Exception:
        needs_grad = False

    if not needs_grad:
        if was_unbatched:
            return result.squeeze(0), converged.squeeze(0)
        return result, converged

    # Make sure result has requires_grad=True for the autograd function
    if not result.requires_grad:
        result = result.clone().requires_grad_(True)

    result = _BroydenImplicitGrad.apply(result, f, was_unbatched)

    if was_unbatched:
        return result.squeeze(0), converged.squeeze(0)
    return result, converged


def broyden(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    method: str = "good",
    jacobian_init: Tensor | None = None,
    xtol: float | None = None,
    rtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> tuple[Tensor, Tensor]:
    """
    Find roots of f(x) = 0 for systems of equations using Broyden's method.

    Broyden's quasi-Newton method approximates the Jacobian matrix from
    iteration history, avoiding the need to compute or estimate the full
    Jacobian at each step. Two variants are available:

    - **"good" (Broyden's first method)**: Updates the Jacobian approximation
      to satisfy the secant equation in the direction of the most recent step.
    - **"bad" (Broyden's second method)**: Updates the inverse Jacobian
      approximation directly, which can be more efficient for large systems.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vector-valued function. Takes tensor of shape ``(B, n)`` or ``(n,)``,
        returns tensor of the same shape. The function should map from R^n
        to R^n.
    x0 : Tensor
        Initial guess for the root. Shape ``(B, n)`` for batch size B and
        system dimension n, or ``(n,)`` for unbatched.
    method : str, default="good"
        Broyden update method:

        - ``"good"``: Broyden's first method (updates Jacobian approximation)
        - ``"bad"``: Broyden's second method (updates inverse Jacobian)
    jacobian_init : Tensor or None, optional
        Initial approximation to the Jacobian matrix. Shape ``(n, n)`` for
        unbatched or ``(B, n, n)`` for batched input. If None (default),
        the identity matrix is used.
    xtol : float, optional
        Absolute tolerance on x change. Convergence requires
        ``max|x_new - x_old| < xtol + rtol * max|x_old|``.
        Default: dtype-aware (1e-3 for float16/bfloat16, 1e-6 for float32,
        1e-12 for float64).
    rtol : float, optional
        Relative tolerance on x change. Combined with xtol for convergence.
        Default: dtype-aware (1e-2 for float16/bfloat16, 1e-5 for float32,
        1e-9 for float64).
    ftol : float, optional
        Tolerance on residual norm. Convergence requires ``max|f(x)| < ftol``.
        Default: dtype-aware (same as xtol).
    maxiter : int, default=100
        Maximum iterations. Non-converged elements will have converged=False.

    Returns
    -------
    tuple[Tensor, Tensor]
        - **root** -- Roots with the same shape as input ``x0``.
          For non-converged elements, this is the best estimate.
        - **converged** -- Boolean tensor of shape ``(B,)`` or scalar for
          unbatched input, indicating which batch elements converged.

    Examples
    --------
    Solve a nonlinear system: x^2 + y^2 = 1, x = y (find point on unit circle
    where x = y):

    >>> import torch
    >>> from torchscience.root_finding import broyden
    >>> def f(x):
    ...     x1, x2 = x[..., 0], x[..., 1]
    ...     f1 = x1**2 + x2**2 - 1  # x^2 + y^2 = 1
    ...     f2 = x1 - x2            # x = y
    ...     return torch.stack([f1, f2], dim=-1)
    >>> x0 = torch.tensor([0.5, 0.5])
    >>> root, converged = broyden(f, x0)
    >>> root  # doctest: +SKIP
    tensor([0.7071, 0.7071])
    >>> converged
    tensor(True)

    Use Broyden's "bad" method for comparison:

    >>> root_bad, converged = broyden(f, x0, method="bad")
    >>> converged
    tensor(True)

    Batched solving (multiple systems in parallel):

    >>> x0 = torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]])
    >>> roots, converged = broyden(f, x0)
    >>> converged.all()
    tensor(True)

    Notes
    -----
    **Broyden's First Method ("good")**: The Jacobian approximation B is
    updated according to:

    .. math::

        B_{k+1} = B_k + \\frac{(\\Delta f_k - B_k \\Delta x_k) \\Delta x_k^T}
                              {\\Delta x_k^T \\Delta x_k}

    where :math:`\\Delta x_k = x_{k+1} - x_k` and
    :math:`\\Delta f_k = f(x_{k+1}) - f(x_k)`.

    **Broyden's Second Method ("bad")**: The inverse Jacobian approximation
    H = B^{-1} is updated according to:

    .. math::

        H_{k+1} = H_k + \\frac{(\\Delta x_k - H_k \\Delta f_k) \\Delta f_k^T H_k}
                              {\\Delta f_k^T H_k \\Delta f_k}

    **Convergence**: Broyden's methods have superlinear convergence near
    roots, which is faster than linear convergence (e.g., fixed-point
    iteration) but slower than the quadratic convergence of Newton's method.
    The advantage is that Broyden's method requires only one function
    evaluation per iteration.

    **Initial Jacobian**: The choice of initial Jacobian approximation can
    affect convergence. The identity matrix is a common default, but providing
    a better approximation (e.g., a finite difference estimate) can improve
    convergence for difficult problems.

    See Also
    --------
    newton_system : Newton's method for systems (uses exact Jacobian)
    fixed_point : Fixed-point iteration
    scipy.optimize.broyden1 : SciPy's Broyden's first method
    scipy.optimize.broyden2 : SciPy's Broyden's second method
    """
    if method not in ("good", "bad"):
        raise ValueError(
            f"method must be 'good' or 'bad', got '{method}'. "
            "'good' is Broyden's first method (updates Jacobian), "
            "'bad' is Broyden's second method (updates inverse Jacobian)."
        )

    # Handle unbatched input
    was_unbatched = x0.dim() == 1
    if was_unbatched:
        x0 = x0.unsqueeze(0)

    batch_size = x0.shape[0]
    n = x0.shape[1]

    if x0.numel() == 0:
        empty_converged = torch.ones(
            batch_size, dtype=torch.bool, device=x0.device
        )
        return _attach_implicit_grad(
            x0.clone(), empty_converged, f, was_unbatched
        )

    x = x0.clone()
    dtype = x.dtype
    device = x.device

    # Get tolerances
    defaults = default_tolerances(dtype)
    if xtol is None:
        xtol = defaults["xtol"]
    if rtol is None:
        rtol = defaults["rtol"]
    if ftol is None:
        ftol = defaults["ftol"]

    # Initialize Jacobian approximation B (or inverse H for "bad" method)
    if jacobian_init is not None:
        if jacobian_init.dim() == 2:
            # Unbatched Jacobian, expand to batch
            B = jacobian_init.unsqueeze(0).expand(batch_size, n, n).clone()
        else:
            B = jacobian_init.clone()
    else:
        # Default to identity matrix
        B = (
            torch.eye(n, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(batch_size, n, n)
            .clone()
        )

    # For "bad" method, B represents H = B^{-1}, so we start with identity
    # (which is its own inverse)

    # Track which batch elements have converged
    converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
    result = x.clone()

    # Evaluate f at initial point
    fx = f(x)

    for _ in range(maxiter):
        # Check for convergence based on function value
        # Use max norm across system dimension
        f_max = torch.abs(fx).max(dim=-1).values  # (B,)
        f_converged = f_max < ftol

        # Solve for step dx
        if method == "good":
            # Solve B @ dx = -f for dx
            try:
                neg_fx = -fx.unsqueeze(-1)  # (B, n, 1)
                dx = torch.linalg.solve(B, neg_fx).squeeze(-1)  # (B, n)
            except RuntimeError:
                # Fallback to pseudoinverse for singular B
                B_pinv = torch.linalg.pinv(B)
                dx = torch.einsum("bij,bj->bi", B_pinv, -fx)
        else:
            # "bad" method: B is H = inverse Jacobian, so dx = -H @ f
            dx = torch.einsum("bij,bj->bi", -B, fx)

        # Newton step
        x_new = x + dx

        # Check convergence based on x change
        dx_max = torch.abs(dx).max(dim=-1).values  # (B,)
        x_max = torch.abs(x).max(dim=-1).values  # (B,)
        x_converged = dx_max < xtol + rtol * x_max

        # Convergence is achieved if either x or f criterion is met
        newly_converged = (x_converged | f_converged) & ~converged

        # Update results for newly converged elements
        result = torch.where(
            newly_converged.unsqueeze(-1).expand_as(x_new), x_new, result
        )
        converged = converged | newly_converged

        if torch.all(converged):
            return _attach_implicit_grad(result, converged, f, was_unbatched)

        # Evaluate f at new point
        fx_new = f(x_new)

        # Compute df = f(x_new) - f(x)
        df = fx_new - fx

        # Update Jacobian approximation using Broyden's formula
        # Only update for non-converged elements
        active = ~converged

        if method == "good":
            # Broyden's first method:
            # B_new = B + ((df - B @ dx) @ dx^T) / (dx^T @ dx)
            Bdx = torch.einsum("bij,bj->bi", B, dx)  # (B, n)
            diff = df - Bdx  # (B, n)

            # dx^T @ dx: (B,)
            dx_dot_dx = torch.einsum("bi,bi->b", dx, dx)

            # Avoid division by zero
            safe_denom = torch.where(
                dx_dot_dx.abs() > 1e-30,
                dx_dot_dx,
                torch.ones_like(dx_dot_dx),
            )

            # Outer product: diff @ dx^T -> (B, n, n)
            outer = torch.einsum("bi,bj->bij", diff, dx)

            # Update: B + outer / denom
            update = outer / safe_denom.unsqueeze(-1).unsqueeze(-1)

            # Only apply update to active (non-converged) elements
            B = torch.where(
                active.unsqueeze(-1).unsqueeze(-1).expand_as(B),
                B + update,
                B,
            )
        else:
            # Broyden's second method (updates inverse Jacobian H):
            # H_new = H + ((dx - H @ df) @ (df^T @ H)) / (df^T @ H @ df)
            Hdf = torch.einsum("bij,bj->bi", B, df)  # (B, n)
            diff = dx - Hdf  # (B, n)

            # df^T @ H: (B, n)
            dfH = torch.einsum("bi,bij->bj", df, B)

            # df^T @ H @ df: (B,)
            denom = torch.einsum("bi,bi->b", df, Hdf)

            # Avoid division by zero
            safe_denom = torch.where(
                denom.abs() > 1e-30,
                denom,
                torch.ones_like(denom),
            )

            # Outer product: diff @ (df^T @ H) -> (B, n, n)
            outer = torch.einsum("bi,bj->bij", diff, dfH)

            # Update: H + outer / denom
            update = outer / safe_denom.unsqueeze(-1).unsqueeze(-1)

            # Only apply update to active (non-converged) elements
            B = torch.where(
                active.unsqueeze(-1).unsqueeze(-1).expand_as(B),
                B + update,
                B,
            )

        # Update x and fx for next iteration (only for unconverged elements)
        x = torch.where(converged.unsqueeze(-1).expand_as(x), x, x_new)
        fx = torch.where(converged.unsqueeze(-1).expand_as(fx), fx, fx_new)

    # Return best estimate for non-converged elements
    result = torch.where(converged.unsqueeze(-1).expand_as(x), result, x)

    return _attach_implicit_grad(result, converged, f, was_unbatched)
