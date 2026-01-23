"""Trust-Region Newton method for systems of equations."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import default_tolerances
from ._differentiation import compute_jacobian


class _TrustRegionImplicitGrad(torch.autograd.Function):
    """Custom autograd for implicit differentiation through Trust-Region Newton.

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
            # We compute: v = -J^{-T} @ grad_output
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
        The root found by Trust-Region Newton. Shape (B, n) or (n,).
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

    result = _TrustRegionImplicitGrad.apply(result, f, was_unbatched)

    if was_unbatched:
        return result.squeeze(0), converged.squeeze(0)
    return result, converged


def _dogleg_step(
    J: Tensor,
    fx: Tensor,
    delta: Tensor,
) -> Tensor:
    """Compute the dogleg step within trust region.

    The dogleg method finds a step that:
    1. Uses the full Newton step if it's within the trust region
    2. Uses a scaled Cauchy (steepest descent) step if the Cauchy point is outside
    3. Interpolates between Cauchy and Newton points otherwise

    Parameters
    ----------
    J : Tensor
        Jacobian matrix. Shape (B, m, n).
    fx : Tensor
        Function values. Shape (B, m).
    delta : Tensor
        Trust region radius. Shape (B,).

    Returns
    -------
    Tensor
        The step to take. Shape (B, n).
    """
    # Compute J^T @ J and J^T @ f
    JtJ = torch.einsum("bmi,bmj->bij", J, J)  # (B, n, n)
    Jtf = torch.einsum("bmi,bm->bi", J, fx)  # (B, n)

    # Compute Newton step: p_N = -solve(J^T J, J^T f) (for least squares)
    # For square systems, this is equivalent to -solve(J, f)
    try:
        neg_Jtf = -Jtf.unsqueeze(-1)  # (B, n, 1)
        p_newton = torch.linalg.solve(JtJ, neg_Jtf).squeeze(-1)  # (B, n)
    except RuntimeError:
        # Fallback to pseudoinverse for singular matrix
        JtJ_pinv = torch.linalg.pinv(JtJ)
        p_newton = torch.einsum("bij,bj->bi", JtJ_pinv, -Jtf)

    # Compute norm of Newton step
    p_newton_norm = torch.linalg.vector_norm(p_newton, dim=-1)  # (B,)

    # If Newton step is within trust region, use it
    use_newton = p_newton_norm <= delta

    # Compute Cauchy (steepest descent) step direction: -J^T f
    # The Cauchy point minimizes the linear model along the gradient direction
    # p_C = -alpha * g where g = J^T f and alpha = ||g||^2 / ||J g||^2
    g = Jtf  # gradient direction (B, n)
    g_norm_sq = torch.einsum("bi,bi->b", g, g)  # ||g||^2 (B,)

    # J @ g gives (B, m)
    Jg = torch.einsum("bmi,bi->bm", J, g)  # (B, m)
    Jg_norm_sq = torch.einsum("bm,bm->b", Jg, Jg)  # ||J g||^2 (B,)

    # Safe division
    safe_Jg_norm_sq = torch.where(
        Jg_norm_sq.abs() > 1e-30,
        Jg_norm_sq,
        torch.ones_like(Jg_norm_sq),
    )
    alpha = g_norm_sq / safe_Jg_norm_sq  # (B,)

    # Cauchy point: p_C = -alpha * g
    p_cauchy = -alpha.unsqueeze(-1) * g  # (B, n)
    p_cauchy_norm = torch.linalg.vector_norm(p_cauchy, dim=-1)  # (B,)

    # If Cauchy point is outside trust region, use scaled Cauchy step
    # Scale to be on the trust region boundary
    use_scaled_cauchy = p_cauchy_norm > delta

    # Scaled Cauchy step: delta * p_cauchy / ||p_cauchy||
    safe_p_cauchy_norm = torch.where(
        p_cauchy_norm > 1e-30,
        p_cauchy_norm,
        torch.ones_like(p_cauchy_norm),
    )
    p_scaled_cauchy = (delta / safe_p_cauchy_norm).unsqueeze(
        -1
    ) * p_cauchy  # (B, n)

    # For the dogleg interpolation case:
    # Find tau in [1, 2] such that ||p_cauchy + (tau-1)*(p_newton - p_cauchy)|| = delta
    # This is a quadratic equation in tau
    d = p_newton - p_cauchy  # (B, n)

    # ||p_C + (tau-1)*d||^2 = delta^2
    # ||p_C||^2 + 2*(tau-1)*<p_C, d> + (tau-1)^2*||d||^2 = delta^2
    # Let s = tau - 1, then:
    # ||d||^2 * s^2 + 2*<p_C, d>*s + (||p_C||^2 - delta^2) = 0

    d_norm_sq = torch.einsum("bi,bi->b", d, d)  # (B,)
    pc_dot_d = torch.einsum("bi,bi->b", p_cauchy, d)  # (B,)
    c = p_cauchy_norm**2 - delta**2  # (B,)

    # Solve quadratic: a*s^2 + b*s + c = 0
    a = d_norm_sq
    b = 2 * pc_dot_d

    # Safe discriminant calculation
    discriminant = b**2 - 4 * a * c
    discriminant = torch.clamp(discriminant, min=0)  # Ensure non-negative

    # Safe division for quadratic formula
    safe_a = torch.where(a.abs() > 1e-30, a, torch.ones_like(a))

    # We want the positive root (s > 0)
    s = (-b + torch.sqrt(discriminant)) / (2 * safe_a)
    s = torch.clamp(s, min=0, max=1)  # tau in [1, 2] means s in [0, 1]

    # Dogleg step: p_C + s * (p_N - p_C)
    p_dogleg = p_cauchy + s.unsqueeze(-1) * d  # (B, n)

    # Select the appropriate step based on conditions
    # Priority: Newton (if inside) > Scaled Cauchy (if Cauchy outside) > Dogleg
    step = torch.where(
        use_newton.unsqueeze(-1).expand_as(p_newton),
        p_newton,
        torch.where(
            use_scaled_cauchy.unsqueeze(-1).expand_as(p_scaled_cauchy),
            p_scaled_cauchy,
            p_dogleg,
        ),
    )

    return step


def trust_region(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    jacobian: Callable[[Tensor], Tensor] | None = None,
    initial_radius: float = 1.0,
    max_radius: float = 100.0,
    xtol: float | None = None,
    rtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 100,
) -> tuple[Tensor, Tensor]:
    """
    Find roots of f(x) = 0 for systems of equations using Trust-Region Newton.

    The Trust-Region Newton method constrains the step size at each iteration
    to remain within a "trust region" where the local linear model is expected
    to be a good approximation. This provides robustness for problems where
    the standard Newton step might be too large or lead to divergence.

    The algorithm minimizes ||f(x)||^2 using a trust-region approach with
    the dogleg method for solving the trust-region subproblem.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vector-valued function. Takes tensor of shape ``(B, n)`` or ``(n,)``,
        returns tensor of shape ``(B, m)`` or ``(m,)``. For square systems,
        m = n.
    x0 : Tensor
        Initial guess for the root. Shape ``(B, n)`` for batch size B and
        system dimension n, or ``(n,)`` for unbatched.
    jacobian : Callable[[Tensor], Tensor], optional
        Explicit Jacobian function. If None (default), the Jacobian is
        computed using autodiff. The function should take input of shape
        ``(B, n)`` and return shape ``(B, m, n)``.
    initial_radius : float, default=1.0
        Initial trust region radius. The step at each iteration is constrained
        to have norm at most this value (initially).
    max_radius : float, default=100.0
        Maximum trust region radius. The radius can grow but never exceeds
        this value.
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
    >>> from torchscience.root_finding import trust_region
    >>> def f(x):
    ...     x1, x2 = x[..., 0], x[..., 1]
    ...     f1 = x1**2 + x2**2 - 1  # x^2 + y^2 = 1
    ...     f2 = x1 - x2            # x = y
    ...     return torch.stack([f1, f2], dim=-1)
    >>> x0 = torch.tensor([0.5, 0.5])
    >>> root, converged = trust_region(f, x0)
    >>> root  # doctest: +SKIP
    tensor([0.7071, 0.7071])
    >>> converged
    tensor(True)

    Batched solving (multiple systems in parallel):

    >>> x0 = torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]])
    >>> roots, converged = trust_region(f, x0)
    >>> converged.all()
    tensor(True)

    Control the trust region size for difficult problems:

    >>> root, converged = trust_region(f, x0, initial_radius=0.5, max_radius=2.0)

    Notes
    -----
    **Algorithm**: At each iteration, the Trust-Region Newton algorithm:

    1. Computes the Jacobian J and current residual f(x)
    2. Computes a candidate step p using the dogleg method within the trust region
    3. Evaluates the actual vs predicted reduction ratio rho
    4. Updates x if rho > threshold (step is accepted)
    5. Adjusts the trust region radius based on rho

    **Trust Region Subproblem**: The dogleg method is used to solve:

    .. math::

        \\min_p \\|f(x) + J p\\|^2 \\quad \\text{s.t.} \\quad \\|p\\| \\le \\Delta

    where :math:`\\Delta` is the trust region radius.

    **Radius Adaptation**: The trust region radius is adjusted based on the
    ratio of actual to predicted reduction:

    - If :math:`\\rho > 0.75` and step is at boundary: increase radius
    - If :math:`\\rho < 0.25`: decrease radius
    - Accept step if :math:`\\rho > 0.1`

    **Convergence**: Trust-region methods have global convergence guarantees
    under mild conditions, making them more robust than pure Newton methods
    for difficult problems or poor initial guesses.

    See Also
    --------
    newton_system : Newton's method for systems (no trust region)
    levenberg_marquardt : Damped least-squares approach
    broyden : Quasi-Newton method with Jacobian approximation
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

    # Initialize trust region radius per batch element
    delta = torch.full(
        (batch_size,), initial_radius, dtype=dtype, device=device
    )

    # Constants for trust region adaptation
    eta = 0.1  # Minimum ratio for accepting step
    rho_good = 0.75  # Threshold to increase radius
    rho_bad = 0.25  # Threshold to decrease radius

    # Evaluate f at initial point
    fx = f(x)
    fx_norm_sq = (fx**2).sum(dim=-1)  # (B,)

    for _ in range(maxiter):
        # Check for convergence based on function value
        f_max = torch.abs(fx).max(dim=-1).values  # (B,)
        f_converged = f_max < ftol

        # Compute Jacobian: (B, m, n)
        J = compute_jacobian(f, x, jacobian=jacobian)

        # Compute step using dogleg method
        p = _dogleg_step(J, fx, delta)

        # Compute predicted reduction
        # predicted = ||f||^2 - ||f + J @ p||^2
        Jp = torch.einsum("bmi,bi->bm", J, p)  # (B, m)
        f_plus_Jp = fx + Jp
        predicted = fx_norm_sq - (f_plus_Jp**2).sum(dim=-1)

        # Compute new point
        x_new = x + p

        # Evaluate f at new point
        fx_new = f(x_new)
        fx_new_norm_sq = (fx_new**2).sum(dim=-1)  # (B,)

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
        accept = rho > eta

        # Check convergence based on x change
        p_max = torch.abs(p).max(dim=-1).values  # (B,)
        x_max = torch.abs(x).max(dim=-1).values  # (B,)
        x_converged = p_max < xtol + rtol * x_max

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

        # Update trust region radius
        # Compute step norm for radius adjustment
        p_norm = torch.linalg.vector_norm(p, dim=-1)  # (B,)

        # If rho > rho_good and step is at trust region boundary, increase radius
        at_boundary = p_norm >= delta * 0.99  # Close to boundary
        increase_radius = (rho > rho_good) & at_boundary

        # If rho < rho_bad, decrease radius
        decrease_radius = rho < rho_bad

        # Apply radius updates
        delta_new = torch.where(
            increase_radius,
            torch.minimum(2.0 * delta, torch.full_like(delta, max_radius)),
            torch.where(
                decrease_radius,
                delta * 0.25,
                delta,
            ),
        )

        # Clamp radius to reasonable range
        delta_new = torch.clamp(delta_new, min=1e-10, max=max_radius)

        # Update for next iteration (only for unconverged elements)
        active = ~converged
        delta = torch.where(active, delta_new, delta)

        # Update x and fx for accepted steps
        x = torch.where((active & accept).unsqueeze(-1).expand_as(x), x_new, x)
        fx = torch.where(
            (active & accept).unsqueeze(-1).expand_as(fx), fx_new, fx
        )
        fx_norm_sq = torch.where(active & accept, fx_new_norm_sq, fx_norm_sq)

    # Return best estimate for non-converged elements
    result = torch.where(converged.unsqueeze(-1).expand_as(x), result, x)

    return _attach_implicit_grad(result, converged, f, was_unbatched)
