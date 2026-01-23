"""Levenberg-Marquardt algorithm for systems of equations."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import default_tolerances
from ._differentiation import compute_jacobian


class _LMImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through Levenberg-Marquardt.

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
        root_batched = root
        grad_batched = grad_output

        batch_size = root_batched.shape[0]
        n = root_batched.shape[1]

        # Compute Jacobian df/dx at the root
        x = root_batched.detach().requires_grad_(True)
        with torch.enable_grad():
            fx = ctx.f_callable(x)

            # Handle case where output dim != input dim (overdetermined systems)
            m = fx.shape[-1]

            # Compute full Jacobian matrix for each batch element
            # J[b, i, j] = d(f_i)/d(x_j) for batch b
            jacobian = torch.zeros(
                batch_size, m, n, dtype=x.dtype, device=x.device
            )
            for i in range(m):
                # Compute gradient of f_i w.r.t. x
                grad_fi = torch.autograd.grad(
                    fx[..., i].sum(),
                    x,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                jacobian[:, i, :] = grad_fi

            # Apply implicit function theorem:
            # For square systems: dL/dtheta = -dL/dx* @ J^{-1} @ df/dtheta
            # For overdetermined: use J^T J (normal equations)
            # We compute: v = -J^{-T} @ grad_output (or pseudo-inverse for non-square)
            try:
                if m == n:
                    # Square system: solve J^T @ v = -grad_output for v
                    neg_grad = -grad_batched.unsqueeze(-1)  # (B, n, 1)
                    v = torch.linalg.solve(
                        jacobian.transpose(-2, -1), neg_grad
                    ).squeeze(-1)  # (B, n)
                else:
                    # Overdetermined: use pseudoinverse
                    J_pinv_T = torch.linalg.pinv(jacobian).transpose(-2, -1)
                    v = torch.einsum("bij,bj->bi", -J_pinv_T, grad_batched)
            except RuntimeError:
                # Fallback to pseudoinverse for singular Jacobian
                J_pinv_T = torch.linalg.pinv(jacobian).transpose(-2, -1)
                v = torch.einsum("bij,bj->bi", -J_pinv_T, grad_batched)

            # Backpropagate v through f to get df/dtheta contribution
            if fx.grad_fn is not None:
                # Need to expand v to match fx shape for overdetermined case
                if m != n:
                    # For overdetermined, we need to compute contribution differently
                    # Use J @ v_x where v_x is what we computed
                    v_expanded = torch.einsum("bij,bj->bi", jacobian, v)
                    torch.autograd.backward(fx, v_expanded)
                else:
                    torch.autograd.backward(fx, v)

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
        The root found by Levenberg-Marquardt. Shape (B, n) or (n,).
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

    result = _LMImplicitGrad.apply(result, f, was_unbatched)

    if was_unbatched:
        return result.squeeze(0), converged.squeeze(0)
    return result, converged


def levenberg_marquardt(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    jacobian: Callable[[Tensor], Tensor] | None = None,
    damping: float = 1e-3,
    xtol: float | None = None,
    rtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> tuple[Tensor, Tensor]:
    """
    Find roots of f(x) = 0 for systems of equations using Levenberg-Marquardt.

    The Levenberg-Marquardt algorithm is a hybrid method that interpolates
    between Gauss-Newton and gradient descent. It is particularly effective
    for nonlinear least-squares problems and systems of equations that may
    have singular or near-singular Jacobians.

    The algorithm solves:
        (J^T J + lambda * I) dx = -J^T f

    where lambda is adaptively adjusted based on the quality of each step.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vector-valued function. Takes tensor of shape ``(B, n)`` or ``(n,)``,
        returns tensor of shape ``(B, m)`` or ``(m,)``. For square systems,
        m = n. Overdetermined systems (m > n) are also supported.
    x0 : Tensor
        Initial guess for the root. Shape ``(B, n)`` for batch size B and
        system dimension n, or ``(n,)`` for unbatched.
    jacobian : Callable[[Tensor], Tensor], optional
        Explicit Jacobian function. If None (default), the Jacobian is
        computed using autodiff. The function should take input of shape
        ``(B, n)`` and return shape ``(B, m, n)``.
    damping : float, default=1e-3
        Initial damping parameter (lambda). Larger values make the algorithm
        more like gradient descent, smaller values more like Gauss-Newton.
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
    >>> from torchscience.root_finding import levenberg_marquardt
    >>> def f(x):
    ...     x1, x2 = x[..., 0], x[..., 1]
    ...     f1 = x1**2 + x2**2 - 1  # x^2 + y^2 = 1
    ...     f2 = x1 - x2            # x = y
    ...     return torch.stack([f1, f2], dim=-1)
    >>> x0 = torch.tensor([0.5, 0.5])
    >>> root, converged = levenberg_marquardt(f, x0)
    >>> root  # doctest: +SKIP
    tensor([0.7071, 0.7071])
    >>> converged
    tensor(True)

    Batched solving (multiple systems in parallel):

    >>> x0 = torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]])
    >>> roots, converged = levenberg_marquardt(f, x0)
    >>> converged.all()
    tensor(True)

    Use a higher initial damping for a difficult problem:

    >>> root, converged = levenberg_marquardt(f, x0, damping=1.0)

    Notes
    -----
    **Algorithm**: At each iteration, the Levenberg-Marquardt algorithm solves:

    .. math::

        (J^T J + \\lambda I) \\Delta x = -J^T f(x)

    where :math:`J` is the Jacobian matrix, :math:`\\lambda` is the damping
    parameter, and :math:`\\Delta x` is the step direction.

    **Damping Adaptation**: The damping parameter is adjusted based on the
    gain ratio:

    .. math::

        \\rho = \\frac{\\|f(x)\\|^2 - \\|f(x + \\Delta x)\\|^2}
                     {\\|f(x)\\|^2 - \\|f(x) + J \\Delta x\\|^2}

    - If :math:`\\rho > 0.75`: decrease :math:`\\lambda` (more Gauss-Newton)
    - If :math:`\\rho < 0.25`: increase :math:`\\lambda` (more gradient descent)

    **Convergence**: The method converges superlinearly near local minima
    with non-singular Jacobian. It is more robust than pure Gauss-Newton
    for problems with singular or near-singular Jacobians.

    **Overdetermined Systems**: For systems with more equations than unknowns
    (m > n), the algorithm finds a least-squares solution.

    See Also
    --------
    newton_system : Newton's method for systems (no damping)
    broyden : Quasi-Newton method with Jacobian approximation
    scipy.optimize.root : SciPy's root finding with 'lm' method
    """
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

    # Track which batch elements have converged
    converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
    result = x.clone()

    # Initialize damping parameter per batch element
    lam = torch.full((batch_size,), damping, dtype=dtype, device=device)

    # Constants for damping adaptation
    rho_good = 0.75  # Threshold to decrease damping
    rho_bad = 0.25  # Threshold to increase damping
    factor_up = 2.0  # Factor to increase damping
    factor_down = 3.0  # Factor to decrease damping

    # Evaluate f at initial point
    fx = f(x)
    fx_norm_sq = (fx**2).sum(dim=-1)  # (B,)

    for _ in range(maxiter):
        # Check for convergence based on function value
        f_max = torch.abs(fx).max(dim=-1).values  # (B,)
        f_converged = f_max < ftol

        # Compute Jacobian: (B, m, n) where m may differ from n
        J = compute_jacobian(f, x, jacobian=jacobian)

        # Compute J^T @ J: (B, n, n)
        JtJ = torch.einsum("bmi,bmj->bij", J, J)

        # Compute J^T @ f: (B, n)
        Jtf = torch.einsum("bmi,bm->bi", J, fx)

        # Solve (J^T J + lambda * I) dx = -J^T f
        # Add damping to diagonal
        eye = torch.eye(n, dtype=dtype, device=device).unsqueeze(
            0
        )  # (1, n, n)
        lam_diag = lam.unsqueeze(-1).unsqueeze(-1) * eye  # (B, n, n)
        A = JtJ + lam_diag

        try:
            neg_Jtf = -Jtf.unsqueeze(-1)  # (B, n, 1)
            dx = torch.linalg.solve(A, neg_Jtf).squeeze(-1)  # (B, n)
        except RuntimeError:
            # Fallback to pseudoinverse for singular matrix
            A_pinv = torch.linalg.pinv(A)
            dx = torch.einsum("bij,bj->bi", A_pinv, -Jtf)

        # Compute new point
        x_new = x + dx

        # Evaluate f at new point
        fx_new = f(x_new)
        fx_new_norm_sq = (fx_new**2).sum(dim=-1)  # (B,)

        # Compute predicted reduction (linearized model)
        # predicted = ||f||^2 - ||f + J @ dx||^2
        #           = ||f||^2 - ||f||^2 - 2 f^T J dx - ||J dx||^2
        #           = -2 f^T J dx - ||J dx||^2
        Jdx = torch.einsum("bmi,bi->bm", J, dx)  # (B, m)
        predicted = -2.0 * torch.einsum("bm,bm->b", fx, Jdx) - (Jdx**2).sum(
            dim=-1
        )

        # Compute actual reduction
        actual = fx_norm_sq - fx_new_norm_sq

        # Compute gain ratio rho = actual / predicted
        # Avoid division by zero
        safe_predicted = torch.where(
            torch.abs(predicted) > 1e-30,
            predicted,
            torch.ones_like(predicted),
        )
        rho = actual / safe_predicted

        # Accept or reject step based on rho
        # Accept if rho > 0 (actual reduction is positive and in same direction
        # as predicted)
        accept = rho > 0

        # Check convergence based on x change
        dx_max = torch.abs(dx).max(dim=-1).values  # (B,)
        x_max = torch.abs(x).max(dim=-1).values  # (B,)
        x_converged = dx_max < xtol + rtol * x_max

        # Convergence is achieved if either x or f criterion is met
        newly_converged = (x_converged | f_converged) & ~converged

        # Update results for newly converged elements
        # Use x_new for accepted steps, x for rejected
        x_final = torch.where(accept.unsqueeze(-1), x_new, x)
        result = torch.where(
            newly_converged.unsqueeze(-1).expand_as(x_final), x_final, result
        )
        converged = converged | newly_converged

        if torch.all(converged):
            return _attach_implicit_grad(result, converged, f, was_unbatched)

        # Update damping parameter
        # Decrease lambda if rho > rho_good (good step)
        # Increase lambda if rho < rho_bad (bad step)
        lam_new = torch.where(
            rho > rho_good,
            lam / factor_down,
            torch.where(rho < rho_bad, lam * factor_up, lam),
        )

        # Clamp damping to reasonable range
        lam_new = torch.clamp(lam_new, min=1e-10, max=1e10)

        # Update for next iteration (only for unconverged and accepted steps)
        active = ~converged
        lam = torch.where(active, lam_new, lam)

        # Update x and fx for accepted steps
        x = torch.where((active & accept).unsqueeze(-1).expand_as(x), x_new, x)
        fx = torch.where(
            (active & accept).unsqueeze(-1).expand_as(fx), fx_new, fx
        )
        fx_norm_sq = torch.where(active & accept, fx_new_norm_sq, fx_norm_sq)

    # Return best estimate for non-converged elements
    result = torch.where(converged.unsqueeze(-1).expand_as(x), result, x)

    return _attach_implicit_grad(result, converged, f, was_unbatched)
