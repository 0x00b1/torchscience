"""Linear programming solver.

Solves problems of the form:

    min  c^T x
    s.t. A_ub x <= b_ub
         A_eq x  = b_eq
         lower <= x <= upper

Uses a primal-dual interior point method after converting to standard form.
"""

from typing import Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult


def _build_standard_form(
    c: Tensor,
    A_ub: Optional[Tensor],
    b_ub: Optional[Tensor],
    A_eq: Optional[Tensor],
    b_eq: Optional[Tensor],
    lower: Optional[Tensor],
    upper: Optional[Tensor],
    n: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert LP to standard form: min c_std^T x_std s.t. A_std x_std = b_std, x_std >= 0.

    Variable layout: x_std = [x_shifted, s_ub, s_upper, x_neg]

    - x_shifted: shifted original variables (x - lower) if lower is finite, else split into pos/neg
    - s_ub: slack variables for inequality constraints
    - s_upper: slack variables for upper bounds
    - x_neg: negative parts for unbounded-below variables

    Returns
    -------
    c_std : Tensor
        Standard form cost vector.
    A_std : Tensor
        Standard form equality constraint matrix.
    b_std : Tensor
        Standard form equality constraint RHS.
    """
    has_ub = A_ub is not None and b_ub is not None
    has_eq = A_eq is not None and b_eq is not None

    # Determine shift from lower bounds
    if lower is not None:
        shift = lower.clone()
        has_lower = torch.isfinite(lower)
    else:
        shift = torch.zeros(n, dtype=dtype, device=device)
        has_lower = torch.zeros(n, dtype=torch.bool, device=device)

    # For variables without finite lower bounds, we use x = x_pos - x_neg
    # where both x_pos >= 0 and x_neg >= 0
    unbounded_below = ~has_lower
    n_free = int(unbounded_below.sum().item())

    # Determine upper bound constraints
    if upper is not None:
        has_upper_finite = torch.isfinite(upper)
    else:
        has_upper_finite = torch.zeros(n, dtype=torch.bool, device=device)
    n_upper = int(has_upper_finite.sum().item())

    m_ub = A_ub.shape[0] if has_ub else 0
    m_eq = A_eq.shape[0] if has_eq else 0

    # Standard form variable count:
    # n (shifted x) + n_free (x_neg for free vars) + m_ub (ub slacks) + n_upper (upper bound slacks)
    n_std = n + n_free + m_ub + n_upper

    # Number of equality constraints:
    # m_eq (original) + m_ub (converted from ineq) + n_upper (upper bound constraints)
    m_std = m_eq + m_ub + n_upper

    c_std = torch.zeros(n_std, dtype=dtype, device=device)
    A_std = torch.zeros(m_std, n_std, dtype=dtype, device=device)
    b_std = torch.zeros(m_std, dtype=dtype, device=device)

    # Cost: c^T x = c^T (x_shifted + shift) = c^T x_shifted + c^T shift
    # For free variables: x_i = x_pos_i - x_neg_i, so cost is c_i * x_pos_i - c_i * x_neg_i
    c_std[:n] = c
    free_idx = 0
    for i in range(n):
        if unbounded_below[i]:
            c_std[n + free_idx] = -c[i]  # x_neg part
            free_idx += 1

    # Slacks for inequality constraints have zero cost (already zero)
    # Slacks for upper bounds have zero cost (already zero)

    row = 0

    # Original equality constraints: A_eq @ x = b_eq
    # A_eq @ (x_shifted + shift) = b_eq
    # A_eq @ x_shifted = b_eq - A_eq @ shift
    # For free variables: A_eq_ij * x_j = A_eq_ij * (x_pos_j - x_neg_j)
    if has_eq:
        A_std[row : row + m_eq, :n] = A_eq
        b_std[row : row + m_eq] = b_eq - A_eq @ shift

        # Adjust for free variables
        free_idx = 0
        for j in range(n):
            if unbounded_below[j]:
                A_std[row : row + m_eq, n + free_idx] = -A_eq[:, j]
                free_idx += 1
        row += m_eq

    # Inequality constraints converted to equality: A_ub @ x + s_ub = b_ub
    # A_ub @ (x_shifted + shift) + s_ub = b_ub
    # A_ub @ x_shifted + s_ub = b_ub - A_ub @ shift
    if has_ub:
        A_std[row : row + m_ub, :n] = A_ub
        # Slack variable columns start at n + n_free
        slack_start = n + n_free
        for i in range(m_ub):
            A_std[row + i, slack_start + i] = 1.0
        b_std[row : row + m_ub] = b_ub - A_ub @ shift

        # Adjust for free variables
        free_idx = 0
        for j in range(n):
            if unbounded_below[j]:
                A_std[row : row + m_ub, n + free_idx] = -A_ub[:, j]
                free_idx += 1
        row += m_ub

    # Upper bound constraints: x_shifted + s_upper = upper - shift (for finite upper bounds)
    if n_upper > 0:
        upper_slack_start = n + n_free + m_ub
        ui = 0
        for j in range(n):
            if has_upper_finite[j]:
                A_std[row, j] = 1.0
                A_std[row, upper_slack_start + ui] = 1.0

                # Adjust for free variables
                if unbounded_below[j]:
                    # Find the free_idx for variable j
                    fidx = int(unbounded_below[:j].sum().item())
                    A_std[row, n + fidx] = -1.0

                ub_val = upper[j] - shift[j]
                b_std[row] = ub_val
                ui += 1
                row += 1

    return c_std, A_std, b_std


def _solve_standard_form_ipm(
    c: Tensor,
    A: Tensor,
    b: Tensor,
    tol: float,
    maxiter: int,
) -> tuple[Tensor, bool, int]:
    """Solve standard form LP: min c^T x s.t. Ax = b, x >= 0.

    Uses Mehrotra predictor-corrector primal-dual interior point method.

    Parameters
    ----------
    c : Tensor
        Cost vector of shape ``(n,)``.
    A : Tensor
        Equality constraint matrix of shape ``(m, n)``.
    b : Tensor
        Equality constraint RHS of shape ``(m,)``.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.

    Returns
    -------
    x : Tensor
        Primal solution.
    converged : bool
        Whether the method converged.
    num_iter : int
        Number of iterations performed.
    """
    m, n = A.shape
    dtype = c.dtype
    device = c.device

    # Initial point: x > 0, s > 0
    # Use a simple heuristic: start near the analytic center
    x = torch.ones(n, dtype=dtype, device=device)
    lam = torch.zeros(m, dtype=dtype, device=device)
    s = torch.ones(n, dtype=dtype, device=device)

    # Try to find a better initial x that approximately satisfies Ax = b
    # Using least-norm solution: x = A^T (A A^T)^{-1} b
    AAT = A @ A.t()
    reg = 1e-8 * torch.eye(m, dtype=dtype, device=device)
    try:
        x_init = A.t() @ torch.linalg.solve(AAT + reg, b)
        # Shift to make positive
        min_x = x_init.min()
        if min_x <= 0:
            x = x_init - min_x + 1.0
        else:
            x = x_init
    except Exception:
        x = torch.ones(n, dtype=dtype, device=device)

    # Initial s from dual feasibility: s = c - A^T lam
    s = c - A.t() @ lam
    min_s = s.min()
    if min_s <= 0:
        s = s - min_s + 1.0

    # Ensure strictly positive
    x = x.clamp(min=1e-4)
    s = s.clamp(min=1e-4)

    converged = False
    num_iter = 0

    for iteration in range(maxiter):
        num_iter = iteration + 1

        # Residuals
        r_dual = A.t() @ lam + s - c  # dual feasibility: A^T lam + s = c
        r_primal = A @ x - b  # primal feasibility: Ax = b
        r_comp = x * s  # complementarity: X S e = 0

        mu = (x * s).sum() / n

        # Check convergence
        r_dual_norm = r_dual.norm(torch.inf).item()
        r_primal_norm = r_primal.norm(torch.inf).item()
        gap = mu.item()

        if r_dual_norm < tol and r_primal_norm < tol and gap < tol:
            converged = True
            break

        # === Predictor step (affine direction) ===
        # Solve:
        #   A dx_aff = -r_primal
        #   A^T dlam_aff + ds_aff = -r_dual
        #   S dx_aff + X ds_aff = -X S e
        #
        # Eliminate ds_aff = -r_dual - A^T dlam_aff
        # Then: S dx_aff + X(-r_dual - A^T dlam_aff) = -X S e
        # => S dx_aff - X A^T dlam_aff = -X S e + X r_dual
        # => dx_aff = S^{-1}(-X S e + X r_dual + X A^T dlam_aff)
        # => dx_aff = -x + (X/S) r_dual + (X/S) A^T dlam_aff
        #
        # Substituting into A dx_aff = -r_primal:
        # A(-x + (X/S) r_dual + (X/S) A^T dlam_aff) = -r_primal
        # -Ax + A(X/S)r_dual + A(X/S)A^T dlam_aff = -r_primal
        # A D^2 A^T dlam_aff = -r_primal + Ax - A D^2 r_dual
        # where D^2 = X/S = diag(x/s)

        d2 = x / s  # X S^{-1} diagonal
        AD = A * d2.unsqueeze(0)  # A @ diag(d2) efficiently
        ADA = AD @ A.t()

        # Regularize for numerical stability
        reg_val = 1e-11 * (1.0 + ADA.diag().abs().max().item())
        ADA_reg = ADA + reg_val * torch.eye(m, dtype=dtype, device=device)

        # RHS for affine direction
        rhs_aff = -r_primal + A @ x - AD @ r_dual

        try:
            dlam_aff = torch.linalg.solve(ADA_reg, rhs_aff)
        except Exception:
            # Fallback: use pseudoinverse
            dlam_aff = ADA_reg.pinverse() @ rhs_aff

        ds_aff = -r_dual - A.t() @ dlam_aff
        dx_aff = -x + d2 * (r_dual + A.t() @ dlam_aff)
        # More directly: dx_aff = d2 * (-s + r_dual + A^T dlam_aff)
        # Actually from the elimination: dx_aff = S^{-1}(-XSe + X r_dual + X A^T dlam_aff)
        # = -x + (x/s)*r_dual + (x/s)*(A^T dlam_aff)
        # Let's recompute more carefully:
        dx_aff = (x / s) * (
            -s - r_comp / x * 0 + r_dual + A.t() @ dlam_aff
        ) - x
        # Simpler: from S dx + X ds = -XSe => dx = S^{-1}(-XSe - X ds)
        dx_aff = (-x * s - x * ds_aff) / s

        # Affine step sizes
        alpha_p_aff = 1.0
        neg_dx = dx_aff < 0
        if neg_dx.any():
            alpha_p_aff = min(1.0, (-x[neg_dx] / dx_aff[neg_dx]).min().item())

        alpha_d_aff = 1.0
        neg_ds = ds_aff < 0
        if neg_ds.any():
            alpha_d_aff = min(1.0, (-s[neg_ds] / ds_aff[neg_ds]).min().item())

        # Compute affine duality gap
        mu_aff = (
            (x + alpha_p_aff * dx_aff) * (s + alpha_d_aff * ds_aff)
        ).sum() / n

        # Centering parameter
        sigma = (mu_aff / mu).clamp(min=0.0, max=1.0) ** 3

        # === Corrector step ===
        # RHS changes: complementarity becomes -XSe + sigma*mu*e - dX_aff * dS_aff * e
        r_comp_cc = x * s - sigma * mu + dx_aff * ds_aff

        # Solve corrected system:
        # dx_cc from: S dx + X ds = -r_comp_cc
        # With the same ADA^T factorization:
        rhs_cc = (
            -r_primal
            + A @ x
            - A @ (d2 * (r_dual))
            + A
            @ (
                (
                    x * r_dual
                    - r_comp_cc
                    + x * (A.t() @ torch.zeros(m, dtype=dtype, device=device))
                )
                / s
            )
        )

        # More carefully: the system with corrector is:
        # A dx = -r_primal
        # A^T dlam + ds = -r_dual
        # S dx + X ds = -r_comp_cc
        #
        # From third: dx = S^{-1}(-r_comp_cc - X ds)
        # From second: ds = -r_dual - A^T dlam
        # So: dx = S^{-1}(-r_comp_cc - X(-r_dual - A^T dlam))
        #       = S^{-1}(-r_comp_cc + X r_dual + X A^T dlam)
        # Substituting into first:
        # A S^{-1}(-r_comp_cc + X r_dual + X A^T dlam) = -r_primal
        # A D^2 A^T dlam = -r_primal + A S^{-1} r_comp_cc - A D^2 r_dual
        rhs_corrected = -r_primal + A @ (r_comp_cc / s) - AD @ r_dual

        try:
            dlam = torch.linalg.solve(ADA_reg, rhs_corrected)
        except Exception:
            dlam = ADA_reg.pinverse() @ rhs_corrected

        ds = -r_dual - A.t() @ dlam
        dx = (-r_comp_cc - x * ds) / s

        # Step sizes with fraction-to-boundary rule
        tau = max(0.995, 1.0 - mu.item())
        alpha_p = 1.0
        neg_dx = dx < 0
        if neg_dx.any():
            alpha_p = min(1.0, (tau * (-x[neg_dx] / dx[neg_dx])).min().item())

        alpha_d = 1.0
        neg_ds = ds < 0
        if neg_ds.any():
            alpha_d = min(1.0, (tau * (-s[neg_ds] / ds[neg_ds])).min().item())

        # Update
        x = x + alpha_p * dx
        lam = lam + alpha_d * dlam
        s = s + alpha_d * ds

        # Safety: ensure positivity
        x = x.clamp(min=1e-14)
        s = s.clamp(min=1e-14)

    return x, converged, num_iter


def linear_program(
    c: Tensor,
    *,
    A_ub: Optional[Tensor] = None,
    b_ub: Optional[Tensor] = None,
    A_eq: Optional[Tensor] = None,
    b_eq: Optional[Tensor] = None,
    bounds: Optional[tuple[Optional[Tensor], Optional[Tensor]]] = None,
    tol: Optional[float] = None,
    maxiter: int = 200,
) -> OptimizeResult:
    r"""Solve a linear program.

    Finds the vector :math:`x` that minimises:

    .. math::

        \min_x \; c^T x

    subject to:

    .. math::

        A_{\mathrm{ub}} x &\le b_{\mathrm{ub}} \\
        A_{\mathrm{eq}} x &= b_{\mathrm{eq}} \\
        \mathrm{lower} &\le x \le \mathrm{upper}

    Uses a primal-dual interior point method after converting to standard form.

    Parameters
    ----------
    c : Tensor
        Linear cost vector of shape ``(n,)``.
    A_ub : Tensor, optional
        Inequality constraint matrix of shape ``(m_ub, n)``.
    b_ub : Tensor, optional
        Inequality constraint bound of shape ``(m_ub,)``.
    A_eq : Tensor, optional
        Equality constraint matrix of shape ``(m_eq, n)``.
    b_eq : Tensor, optional
        Equality constraint bound of shape ``(m_eq,)``.
    bounds : tuple of (Tensor or None, Tensor or None), optional
        ``(lower, upper)`` bounds on the decision variables. Each is a
        tensor of shape ``(n,)`` or ``None`` (meaning no bound in that
        direction). Default: no bounds.
    tol : float, optional
        Convergence tolerance. Default: ``eps^{1/3}`` for the input dtype.
    maxiter : int, optional
        Maximum number of interior-point iterations. Default: 200.

    Returns
    -------
    OptimizeResult
        Named tuple with fields:

        - ``x``: Primal solution of shape ``(n,)``.
        - ``converged``: Boolean convergence indicator.
        - ``num_iterations``: Iteration count (``int64``).
        - ``fun``: Objective value at the solution.

    Examples
    --------
    Minimise a linear objective with inequality constraints:

    >>> c = torch.tensor([-1.0, -2.0])
    >>> A_eq = torch.tensor([[1.0, 1.0]])
    >>> b_eq = torch.tensor([1.0])
    >>> result = linear_program(c, A_eq=A_eq, b_eq=b_eq, bounds=(torch.zeros(2), None))
    >>> result.x  # approximately [0, 1]
    tensor([..., ...])

    References
    ----------
    - Wright, S.J. *Primal-Dual Interior-Point Methods*, SIAM, 1997.
    - Nocedal, J. and Wright, S.J. *Numerical Optimization*, Chapter 14.
    """
    dtype = c.dtype
    device = c.device
    n = c.shape[0]

    if tol is None:
        tol = float(torch.finfo(dtype).eps ** (1.0 / 3.0))

    # Parse bounds
    lower: Optional[Tensor] = None
    upper: Optional[Tensor] = None
    if bounds is not None:
        lower, upper = bounds
        if lower is not None:
            lower = lower.to(dtype=dtype, device=device)
        if upper is not None:
            upper = upper.to(dtype=dtype, device=device)

    # Build standard form
    c_std, A_std, b_std = _build_standard_form(
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        lower,
        upper,
        n,
        dtype,
        device,
    )

    # Solve standard form LP
    x_std, converged, num_iter = _solve_standard_form_ipm(
        c_std,
        A_std,
        b_std,
        tol,
        maxiter,
    )

    # Extract original variables from standard form solution
    # x_std[:n] contains the shifted original variables
    x_shifted = x_std[:n]

    # Determine shift
    if lower is not None:
        shift = lower.clone()
        has_lower = torch.isfinite(lower)
        shift[~has_lower] = 0.0
    else:
        shift = torch.zeros(n, dtype=dtype, device=device)
        has_lower = torch.zeros(n, dtype=torch.bool, device=device)

    unbounded_below = ~has_lower
    n_free = int(unbounded_below.sum().item())

    # For free variables, x = x_pos - x_neg
    x_out = x_shifted + shift
    if n_free > 0:
        x_neg = x_std[n : n + n_free]
        free_idx = 0
        for j in range(n):
            if unbounded_below[j]:
                x_out[j] = x_shifted[j] - x_neg[free_idx] + shift[j]
                free_idx += 1

    # Compute objective value
    fun = c @ x_out

    return OptimizeResult(
        x=x_out.detach(),
        converged=torch.tensor(converged, device=device),
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=device
        ),
        fun=fun.detach(),
    )
