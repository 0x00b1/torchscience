"""Newton-Raphson method for systems of equations."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import default_tolerances
from ._differentiation import compute_jacobian


def newton_system(
    f: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    jacobian: Callable[[Tensor], Tensor] | None = None,
    xtol: float | None = None,
    rtol: float | None = None,
    ftol: float | None = None,
    maxiter: int = 50,
) -> tuple[Tensor, Tensor]:
    """
    Find roots of f(x) = 0 for systems of equations using Newton-Raphson method.

    Newton's method for systems uses the iteration:
        J(x_n) @ dx = -f(x_n)
        x_{n+1} = x_n + dx

    where J(x) is the Jacobian matrix of f at x.

    Parameters
    ----------
    f : Callable[[Tensor], Tensor]
        Vector-valued function. Takes tensor of shape ``(B, n)`` or ``(n,)``,
        returns tensor of the same shape. The function should be differentiable
        for autodiff to compute the Jacobian.
    x0 : Tensor
        Initial guess for the root. Shape ``(B, n)`` for batch size B and
        system dimension n, or ``(n,)`` for unbatched.
    jacobian : Callable[[Tensor], Tensor], optional
        Explicit Jacobian function. If None (default), the Jacobian is
        computed using autodiff. The function should take input of shape
        ``(B, n)`` and return shape ``(B, n, n)``.
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
    maxiter : int, default=50
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
    >>> from torchscience.root_finding import newton_system
    >>> def f(x):
    ...     x1, x2 = x[..., 0], x[..., 1]
    ...     f1 = x1**2 + x2**2 - 1  # x^2 + y^2 = 1
    ...     f2 = x1 - x2            # x = y
    ...     return torch.stack([f1, f2], dim=-1)
    >>> x0 = torch.tensor([0.5, 0.5])
    >>> root, converged = newton_system(f, x0)
    >>> root  # doctest: +SKIP
    tensor([0.7071, 0.7071])
    >>> converged
    tensor(True)

    Solve a linear system Ax = b (x1 + x2 = 3, 2*x1 - x2 = 0):

    >>> def f(x):
    ...     x1, x2 = x[..., 0], x[..., 1]
    ...     f1 = x1 + x2 - 3     # x + y = 3
    ...     f2 = 2*x1 - x2       # 2x - y = 0
    ...     return torch.stack([f1, f2], dim=-1)
    >>> x0 = torch.tensor([0.0, 0.0])
    >>> root, converged = newton_system(f, x0)
    >>> root  # doctest: +SKIP
    tensor([1.0000, 2.0000])

    Batched solving (multiple systems in parallel):

    >>> # Solve x^2 - c = 0 for c = [2, 3, 4] (1D systems batched)
    >>> c = torch.tensor([[2.0], [3.0], [4.0]])
    >>> f = lambda x: x**2 - c
    >>> x0 = torch.tensor([[1.5], [1.5], [1.5]])
    >>> roots, converged = newton_system(f, x0)
    >>> roots  # doctest: +SKIP
    tensor([[1.4142], [1.7321], [2.0000]])

    Notes
    -----
    **Convergence**: Newton's method has quadratic convergence near simple
    roots when the Jacobian is non-singular. It can fail to converge when:

    - The initial guess is far from the root
    - The Jacobian is singular or nearly singular at some iteration
    - The system has no solution or multiple solutions

    **Singular Jacobian Handling**: When ``torch.linalg.solve`` fails due to
    a singular Jacobian, the method falls back to using the pseudoinverse
    via ``torch.linalg.pinv``. This allows iterations to continue but may
    result in slower convergence.

    **Batch Dimension**: The first dimension is treated as the batch dimension.
    Each batch element is solved independently.

    See Also
    --------
    newton : Newton's method for scalar functions
    scipy.optimize.fsolve : SciPy's general nonlinear solver
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
        if was_unbatched:
            return x0.squeeze(0), empty_converged.squeeze(0)
        return x0.clone(), empty_converged

    x = x0.clone()

    # Get tolerances
    dtype = x.dtype
    defaults = default_tolerances(dtype)
    if xtol is None:
        xtol = defaults["xtol"]
    if rtol is None:
        rtol = defaults["rtol"]
    if ftol is None:
        ftol = defaults["ftol"]

    # Track which batch elements have converged
    converged = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    result = x.clone()

    for _ in range(maxiter):
        # Evaluate function: (B, n)
        fx = f(x)

        # Check for convergence based on function value
        # Use max norm across system dimension
        f_max = torch.abs(fx).max(dim=-1).values  # (B,)
        f_converged = f_max < ftol

        # Compute Jacobian: (B, n, n)
        J = compute_jacobian(f, x, jacobian=jacobian)

        # Solve J @ dx = -fx for each batch element
        # dx has shape (B, n)
        try:
            # Add batch dimension to fx for solve: (B, n, 1)
            neg_fx = -fx.unsqueeze(-1)
            # Solve: (B, n, n) @ (B, n, 1) -> (B, n, 1)
            dx = torch.linalg.solve(J, neg_fx).squeeze(-1)
        except RuntimeError:
            # Fallback to pseudoinverse for singular Jacobians
            # J_pinv: (B, n, n), -fx: (B, n)
            J_pinv = torch.linalg.pinv(J)
            dx = torch.einsum("bij,bj->bi", J_pinv, -fx)

        # Newton step
        x_new = x + dx

        # Check convergence based on x change
        # Use max norm across system dimension
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
            if was_unbatched:
                return result.squeeze(0), converged.squeeze(0)
            return result, converged

        # Update x for next iteration (only unconverged elements)
        x = torch.where(converged.unsqueeze(-1).expand_as(x), x, x_new)

    # Return best estimate for non-converged elements
    result = torch.where(converged.unsqueeze(-1).expand_as(x), result, x)

    if was_unbatched:
        return result.squeeze(0), converged.squeeze(0)
    return result, converged
