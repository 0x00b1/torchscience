"""Quadratic programming solver.

Solves problems of the form:

    min  0.5 * x^T Q x + c^T x
    s.t. A_ub x <= b_ub
         A_eq x  = b_eq

Uses a direct KKT solve for equality-only problems and a primal-dual
interior point method when inequality constraints are present.
"""

from typing import Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult


class _QPImplicitGrad(torch.autograd.Function):
    """Implicit differentiation through the QP solution.

    At the optimum the KKT conditions hold:

        Q x* + c + A_eq^T nu* + A_ub^T lambda* = 0

    Differentiating these stationarity conditions with respect to the
    problem data (Q, c, A_eq, b_eq, A_ub, b_ub) and applying the
    implicit function theorem yields gradients that flow through the
    QP solution.
    """

    @staticmethod
    def forward(
        ctx,
        x_opt: Tensor,
        Q: Tensor,
        c: Tensor,
        A_eq: Optional[Tensor],
        b_eq: Optional[Tensor],
        A_ub: Optional[Tensor],
        b_ub: Optional[Tensor],
        nu: Optional[Tensor],
        lam: Optional[Tensor],
    ) -> Tensor:
        ctx.save_for_backward(x_opt, Q, c, A_eq, b_eq, A_ub, b_ub, nu, lam)
        return x_opt.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x_opt, Q, c, A_eq, b_eq, A_ub, b_ub, nu, lam = ctx.saved_tensors
        n = x_opt.shape[0]

        # Build the KKT matrix for the active constraints at the solution.
        # For the unconstrained or equality-only case the KKT system is:
        #   [Q   A_eq^T] [dx]   = [-dc]
        #   [A_eq  0   ] [dnu]    [-db_eq]
        #
        # For inequality constraints we linearise around the active set.
        # Active inequalities are those where lambda* > 0.

        blocks_top = [Q]
        blocks_bot_rows = []
        rhs_top = grad_output

        m_eq = 0
        if A_eq is not None:
            m_eq = A_eq.shape[0]
            blocks_top.append(A_eq.t())

        m_active = 0
        active_mask = None
        if lam is not None and A_ub is not None:
            active_mask = lam > 1e-8
            m_active = int(active_mask.sum().item())
            if m_active > 0:
                A_active = A_ub[active_mask]
                blocks_top.append(A_active.t())

        # Assemble full KKT matrix
        kkt_size = n + m_eq + m_active
        KKT = torch.zeros(kkt_size, kkt_size, dtype=Q.dtype, device=Q.device)
        KKT[:n, :n] = Q

        col = n
        if m_eq > 0:
            KKT[:n, col : col + m_eq] = A_eq.t()
            KKT[col : col + m_eq, :n] = A_eq
            col += m_eq

        if m_active > 0:
            A_active = A_ub[active_mask]
            KKT[:n, col : col + m_active] = A_active.t()
            KKT[col : col + m_active, :n] = A_active
            col += m_active

        # Regularise
        reg = 1e-8 * torch.eye(kkt_size, dtype=Q.dtype, device=Q.device)
        KKT = KKT + reg

        rhs = torch.zeros(kkt_size, dtype=Q.dtype, device=Q.device)
        rhs[:n] = grad_output

        # Solve KKT^T v = rhs  (KKT is symmetric so KKT^T = KKT)
        v = torch.linalg.solve(KKT, rhs)
        dx = v[:n]

        # Gradients w.r.t. problem data
        grad_Q = None
        grad_c = None
        grad_A_eq = None
        grad_b_eq = None
        grad_A_ub = None
        grad_b_ub = None

        if c.requires_grad:
            # From stationarity: Q x* + c + ... = 0
            # Differentiating: Q dx* + dc = 0  =>  dx*/dc = -Q^{-1}
            # So d(loss)/dc = (dx*/dc)^T grad_output = -Q^{-1} grad_output = -dx
            grad_c = -dx

        if Q.requires_grad:
            grad_Q = -0.5 * (torch.outer(dx, x_opt) + torch.outer(x_opt, dx))

        if A_eq is not None and A_eq.requires_grad:
            v_eq = v[n : n + m_eq]
            grad_A_eq = -(
                torch.outer(v_eq, x_opt) + torch.outer(nu, dx)
                if nu is not None
                else torch.outer(v_eq, x_opt)
            )

        if b_eq is not None and b_eq.requires_grad:
            grad_b_eq = v[n : n + m_eq]

        if A_ub is not None and A_ub.requires_grad and m_active > 0:
            v_ineq = v[n + m_eq : n + m_eq + m_active]
            lam_active = lam[active_mask]
            grad_A_ub_active = -(
                torch.outer(v_ineq, x_opt) + torch.outer(lam_active, dx)
            )
            grad_A_ub = torch.zeros_like(A_ub)
            grad_A_ub[active_mask] = grad_A_ub_active

        if b_ub is not None and b_ub.requires_grad and m_active > 0:
            grad_b_ub = torch.zeros_like(b_ub)
            grad_b_ub[active_mask] = v[n + m_eq : n + m_eq + m_active]

        return (
            dx,
            grad_Q,
            grad_c,
            grad_A_eq,
            grad_b_eq,
            grad_A_ub,
            grad_b_ub,
            None,
            None,
        )


def _solve_unconstrained(Q: Tensor, c: Tensor) -> Tensor:
    """Solve unconstrained QP: x* = -Q^{-1} c."""
    return torch.linalg.solve(Q, -c)


def _solve_equality_kkt(
    Q: Tensor,
    c: Tensor,
    A_eq: Tensor,
    b_eq: Tensor,
) -> tuple[Tensor, Tensor]:
    """Solve equality-constrained QP via the KKT system.

    The KKT system is:

        [Q    A_eq^T] [x ]   [-c   ]
        [A_eq  0    ] [nu] = [ b_eq ]

    Returns
    -------
    x : Tensor
        Primal solution of shape ``(n,)``.
    nu : Tensor
        Dual variables for equality constraints of shape ``(m_eq,)``.
    """
    n = Q.shape[0]
    m_eq = A_eq.shape[0]
    kkt_size = n + m_eq

    KKT = torch.zeros(kkt_size, kkt_size, dtype=Q.dtype, device=Q.device)
    KKT[:n, :n] = Q
    KKT[:n, n:] = A_eq.t()
    KKT[n:, :n] = A_eq

    rhs = torch.zeros(kkt_size, dtype=Q.dtype, device=Q.device)
    rhs[:n] = -c
    rhs[n:] = b_eq

    # Add small regularisation to bottom-right block for numerical stability
    KKT[n:, n:] -= 1e-12 * torch.eye(m_eq, dtype=Q.dtype, device=Q.device)

    sol = torch.linalg.solve(KKT, rhs)
    x = sol[:n]
    nu = sol[n:]
    return x, nu


def _solve_interior_point(
    Q: Tensor,
    c: Tensor,
    A_ub: Tensor,
    b_ub: Tensor,
    A_eq: Optional[Tensor],
    b_eq: Optional[Tensor],
    tol: float,
    maxiter: int,
) -> tuple[Tensor, Optional[Tensor], Tensor]:
    """Primal-dual interior point method for QP with inequality constraints.

    Solves the KKT system:
        Q x + c + A_eq^T nu + A_ub^T lambda = 0
        A_eq x = b_eq
        A_ub x + s = b_ub
        lambda * s = mu e  (complementarity)
        s >= 0, lambda >= 0

    Parameters
    ----------
    Q, c : Tensor
        Quadratic and linear cost.
    A_ub, b_ub : Tensor
        Inequality constraints A_ub x <= b_ub.
    A_eq, b_eq : Tensor or None
        Equality constraints A_eq x = b_eq.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum number of iterations.

    Returns
    -------
    x : Tensor
        Primal solution.
    nu : Tensor or None
        Equality dual variables.
    lam : Tensor
        Inequality dual variables.
    """
    n = Q.shape[0]
    m_ub = A_ub.shape[0]
    m_eq = A_eq.shape[0] if A_eq is not None else 0
    dtype = Q.dtype
    device = Q.device

    # Initialise primal variable: try unconstrained solution, then project
    # to feasibility.
    x = torch.zeros(n, dtype=dtype, device=device)

    # Slack variables s = b_ub - A_ub @ x (must be positive)
    s = b_ub - A_ub @ x
    min_s = s.min()
    if min_s <= 0:
        # Shift x so that s > 0
        x = x - (A_ub.t() @ torch.clamp(-s + 1.0, min=0.0)) / (
            (A_ub * A_ub).sum() + 1e-12
        )
        s = b_ub - A_ub @ x
        min_s = s.min()
        if min_s <= 0:
            s = s - min_s + 1.0

    # Dual variables
    lam = torch.ones(m_ub, dtype=dtype, device=device)
    nu = torch.zeros(m_eq, dtype=dtype, device=device) if m_eq > 0 else None

    for iteration in range(maxiter):
        # Compute residuals
        # r_dual = Q x + c + A_ub^T lam [+ A_eq^T nu]
        r_dual = Q @ x + c + A_ub.t() @ lam
        if m_eq > 0 and nu is not None:
            r_dual = r_dual + A_eq.t() @ nu

        # r_primal_ub = A_ub x + s - b_ub
        r_primal_ub = A_ub @ x + s - b_ub

        # r_primal_eq = A_eq x - b_eq (if applicable)
        r_primal_eq = None
        if m_eq > 0:
            r_primal_eq = A_eq @ x - b_eq

        # r_comp = lam * s - mu * e (complementarity)
        mu = (lam * s).sum() / m_ub
        sigma = 0.3  # centering parameter
        r_comp = lam * s - sigma * mu

        # Check convergence
        res_norm = r_dual.abs().max().item()
        res_norm = max(res_norm, r_primal_ub.abs().max().item())
        if r_primal_eq is not None:
            res_norm = max(res_norm, r_primal_eq.abs().max().item())
        comp_norm = (lam * s).abs().max().item()

        if res_norm < tol and comp_norm < tol:
            break

        # Build and solve the Newton system.
        # Eliminating ds from the complementarity equation:
        #   ds = -(r_comp + s * dlam) / lam  ... wrong, let's use standard form.
        #
        # The full system is:
        #   [Q        0   A_ub^T  (A_eq^T)] [dx  ]   [-r_dual      ]
        #   [A_ub     I   0       0       ] [ds  ] = [-r_primal_ub ]
        #   [0     Lam    S       0       ] [dlam]   [-r_comp       ]
        #   [(A_eq   0    0       0)      ] [dnu ]   [(-r_primal_eq)]
        #
        # We reduce this by eliminating ds and dlam:
        #   ds = -r_primal_ub - A_ub dx
        #   From complementarity row: Lam ds + S dlam = -r_comp
        #     => S dlam = -r_comp - Lam ds = -r_comp - Lam(-r_primal_ub - A_ub dx)
        #     => dlam = S^{-1} (-r_comp + Lam r_primal_ub + Lam A_ub dx)
        #
        # Substituting into the dual residual:
        #   Q dx + A_ub^T dlam [+ A_eq^T dnu] = -r_dual
        #   Q dx + A_ub^T S^{-1}(-r_comp + Lam r_primal_ub + Lam A_ub dx) [+ A_eq^T dnu] = -r_dual
        #   (Q + A_ub^T S^{-1} Lam A_ub) dx [+ A_eq^T dnu] = -r_dual - A_ub^T S^{-1}(-r_comp + Lam r_primal_ub)
        #
        # Let D = S^{-1} Lam (diagonal, element-wise lam/s)
        D = lam / s  # shape (m_ub,)

        # Reduced system:
        # [Q + A_ub^T D A_ub   A_eq^T ] [dx ]   [rhs_x  ]
        # [A_eq                 0      ] [dnu] = [rhs_eq ]

        Q_bar = Q + A_ub.t() @ (D.unsqueeze(1) * A_ub)
        rhs_x = -r_dual - A_ub.t() @ ((-r_comp + lam * r_primal_ub) / s)

        if m_eq > 0:
            kkt_size = n + m_eq
            KKT = torch.zeros(kkt_size, kkt_size, dtype=dtype, device=device)
            KKT[:n, :n] = Q_bar
            KKT[:n, n:] = A_eq.t()
            KKT[n:, :n] = A_eq
            # Regularise bottom-right
            KKT[n:, n:] -= 1e-12 * torch.eye(m_eq, dtype=dtype, device=device)

            rhs = torch.zeros(kkt_size, dtype=dtype, device=device)
            rhs[:n] = rhs_x
            rhs[n:] = -r_primal_eq

            sol = torch.linalg.solve(KKT, rhs)
            dx = sol[:n]
            dnu = sol[n:]
        else:
            # Add regularisation
            reg = 1e-12 * torch.eye(n, dtype=dtype, device=device)
            dx = torch.linalg.solve(Q_bar + reg, rhs_x)
            dnu = None

        # Recover ds, dlam
        ds = -r_primal_ub - A_ub @ dx
        dlam = (-r_comp + lam * r_primal_ub + lam * (A_ub @ dx)) / s
        # More precisely: dlam = S^{-1}(-r_comp - Lam ds) but the above is equivalent
        # Let's use the direct formula:
        dlam = (-r_comp - lam * ds) / s

        # Step size: ensure s + alpha * ds > 0 and lam + alpha * dlam > 0
        alpha_p = 1.0
        alpha_d = 1.0
        tau = 0.995  # fraction-to-boundary

        neg_ds = ds < 0
        if neg_ds.any():
            alpha_p = min(
                alpha_p,
                (tau * (-s[neg_ds] / ds[neg_ds])).min().item(),
            )

        neg_dlam = dlam < 0
        if neg_dlam.any():
            alpha_d = min(
                alpha_d,
                (tau * (-lam[neg_dlam] / dlam[neg_dlam])).min().item(),
            )

        # Update
        x = x + alpha_p * dx
        s = s + alpha_p * ds
        lam = lam + alpha_d * dlam
        if dnu is not None and nu is not None:
            nu = nu + alpha_d * dnu

        # Safety clamp
        s = s.clamp(min=1e-14)
        lam = lam.clamp(min=1e-14)

    return x, nu, lam


def quadratic_program(
    Q: Tensor,
    c: Tensor,
    *,
    A_ub: Optional[Tensor] = None,
    b_ub: Optional[Tensor] = None,
    A_eq: Optional[Tensor] = None,
    b_eq: Optional[Tensor] = None,
    tol: Optional[float] = None,
    maxiter: int = 200,
) -> OptimizeResult:
    r"""Solve a convex quadratic program.

    Finds the vector :math:`x` that minimises:

    .. math::

        \min_x \; \tfrac{1}{2} x^T Q x + c^T x

    subject to:

    .. math::

        A_{\mathrm{ub}} x &\le b_{\mathrm{ub}} \\
        A_{\mathrm{eq}} x &= b_{\mathrm{eq}}

    The matrix :math:`Q` must be symmetric positive semi-definite.

    Parameters
    ----------
    Q : Tensor
        Symmetric positive semi-definite cost matrix of shape ``(n, n)``.
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
    tol : float, optional
        Convergence tolerance. Default: ``sqrt(eps)`` for the input dtype.
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
    Unconstrained QP with identity Hessian:

    >>> Q = torch.eye(2)
    >>> c = torch.tensor([-1.0, -2.0])
    >>> result = quadratic_program(Q, c)
    >>> result.x
    tensor([1., 2.])

    With an equality constraint:

    >>> result = quadratic_program(
    ...     torch.eye(2),
    ...     torch.zeros(2),
    ...     A_eq=torch.tensor([[1.0, 1.0]]),
    ...     b_eq=torch.tensor([1.0]),
    ... )
    >>> result.x
    tensor([0.5000, 0.5000])

    References
    ----------
    - Nocedal, J. and Wright, S.J. *Numerical Optimization*, Chapter 16.
    - Boyd, S. and Vandenberghe, L. *Convex Optimization*, Chapter 11.
    """
    dtype = Q.dtype
    device = Q.device

    if tol is None:
        tol = float(torch.finfo(dtype).eps ** 0.5)

    # Detach inputs for the forward solve; gradients flow via implicit diff.
    Q_d = Q.detach()
    c_d = c.detach()
    A_eq_d = A_eq.detach() if A_eq is not None else None
    b_eq_d = b_eq.detach() if b_eq is not None else None
    A_ub_d = A_ub.detach() if A_ub is not None else None
    b_ub_d = b_ub.detach() if b_ub is not None else None

    has_ub = A_ub is not None and b_ub is not None
    has_eq = A_eq is not None and b_eq is not None

    nu: Optional[Tensor] = None
    lam: Optional[Tensor] = None
    num_iter = 0

    if not has_ub and not has_eq:
        # Unconstrained
        x = _solve_unconstrained(Q_d, c_d)
        converged = torch.tensor(True, device=device)
        num_iter = 1
    elif not has_ub and has_eq:
        # Equality-only
        x, nu = _solve_equality_kkt(Q_d, c_d, A_eq_d, b_eq_d)
        converged = torch.tensor(True, device=device)
        num_iter = 1
    else:
        # Interior point for inequality (and possibly equality) constraints
        x, nu, lam = _solve_interior_point(
            Q_d, c_d, A_ub_d, b_ub_d, A_eq_d, b_eq_d, tol, maxiter
        )
        # Check convergence: primal feasibility + stationarity
        r_dual = Q_d @ x + c_d
        if A_ub_d is not None:
            r_dual = r_dual + A_ub_d.t() @ lam
        if A_eq_d is not None and nu is not None:
            r_dual = r_dual + A_eq_d.t() @ nu
        res = r_dual.abs().max().item()
        if A_ub_d is not None:
            ub_viol = (A_ub_d @ x - b_ub_d).clamp(min=0).max().item()
            res = max(res, ub_viol)
        if A_eq_d is not None:
            eq_viol = (A_eq_d @ x - b_eq_d).abs().max().item()
            res = max(res, eq_viol)
        converged = torch.tensor(res < tol * 100, device=device)
        num_iter = maxiter  # conservative

    # Objective value
    with torch.no_grad():
        fun = 0.5 * x @ Q_d @ x + c_d @ x

    # Attach implicit gradients through autograd
    # We need to pass tensors (not None) to save_for_backward.
    # Use dummy tensors for missing constraints.
    _A_eq = (
        A_eq
        if A_eq is not None
        else torch.zeros(0, Q.shape[0], dtype=dtype, device=device)
    )
    _b_eq = (
        b_eq
        if b_eq is not None
        else torch.zeros(0, dtype=dtype, device=device)
    )
    _A_ub = (
        A_ub
        if A_ub is not None
        else torch.zeros(0, Q.shape[0], dtype=dtype, device=device)
    )
    _b_ub = (
        b_ub
        if b_ub is not None
        else torch.zeros(0, dtype=dtype, device=device)
    )
    _nu = nu if nu is not None else torch.zeros(0, dtype=dtype, device=device)
    _lam = (
        lam if lam is not None else torch.zeros(0, dtype=dtype, device=device)
    )

    x_out = _QPImplicitGrad.apply(
        x,
        Q,
        c,
        _A_eq,
        _b_eq,
        _A_ub,
        _b_ub,
        _nu,
        _lam,
    )

    return OptimizeResult(
        x=x_out,
        converged=converged,
        num_iterations=torch.tensor(
            num_iter, dtype=torch.int64, device=device
        ),
        fun=fun,
    )
