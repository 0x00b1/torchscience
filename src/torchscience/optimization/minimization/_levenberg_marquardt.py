from typing import Callable, Optional

import torch
from torch import Tensor


class _LMImplicitGrad(torch.autograd.Function):
    """
    Implicit differentiation through Levenberg-Marquardt optimum.

    At the optimum x*, the gradient of the loss is zero: J^T r = 0
    where J is the Jacobian of residuals and r is the residual vector.

    Using implicit differentiation:
        d(J^T r)/dx * dx/dθ + d(J^T r)/dθ = 0

    For least squares, this simplifies to:
        dx*/dθ = -(J^T J)^{-1} J^T (∂r/∂θ)
    """

    @staticmethod
    def forward(
        ctx,
        result: Tensor,
        residuals_callable: Callable[[Tensor], Tensor],
    ) -> Tensor:
        ctx.residuals_callable = residuals_callable
        ctx.save_for_backward(result)
        return result.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (result,) = ctx.saved_tensors

        with torch.enable_grad():
            x = result.detach().requires_grad_(True)
            r = ctx.residuals_callable(x)

            # Compute Jacobian J = ∂r/∂x
            J = torch.func.jacrev(ctx.residuals_callable)(x)

            # Ensure J is 2D: (num_residuals, num_params)
            if J.dim() == 1:
                J = J.unsqueeze(0)

            # Solve (J^T J) v = J^T grad_output for implicit gradient
            # grad_output is the gradient w.r.t. x*, shape (num_params,)
            JtJ = J.T @ J
            Jt_grad = J.T @ grad_output

            # Add regularization for numerical stability
            reg = 1e-6 * torch.eye(
                JtJ.shape[0], dtype=JtJ.dtype, device=JtJ.device
            )
            v = torch.linalg.solve(JtJ + reg, Jt_grad)

            # Backpropagate through residuals with -v as the gradient
            # This computes -v^T (∂r/∂θ) which gives dx*/dθ
            r.backward(-J @ v)

        return None, None


def levenberg_marquardt(
    residuals: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    jacobian: Optional[Callable[[Tensor], Tensor]] = None,
    tol: Optional[float] = None,
    maxiter: int = 100,
    damping: float = 1e-3,
) -> Tensor:
    r"""
    Levenberg-Marquardt algorithm for nonlinear least squares.

    Finds parameters x that minimize the sum of squared residuals:

    .. math::

        \min_x \|r(x)\|^2 = \min_x \sum_i r_i(x)^2

    The algorithm interpolates between Gauss-Newton (fast near optimum)
    and gradient descent (robust far from optimum) using an adaptive
    damping parameter.

    Parameters
    ----------
    residuals : Callable[[Tensor], Tensor]
        Residual function. Takes parameters of shape ``(n,)`` and returns
        residuals of shape ``(m,)`` where ``m >= n``.
    x0 : Tensor
        Initial parameter guess of shape ``(n,)``.
    jacobian : Callable, optional
        Jacobian of residuals. If None, computed via ``torch.func.jacrev``.
        Should return a tensor of shape ``(m, n)``.
    tol : float, optional
        Convergence tolerance on gradient norm. Default: ``sqrt(eps)`` for dtype.
    maxiter : int
        Maximum number of iterations. Default: 100.
    damping : float
        Initial Levenberg-Marquardt damping parameter. Default: 1e-3.

    Returns
    -------
    Tensor
        Optimized parameters of shape ``(n,)``.

    Examples
    --------
    Fit a line y = ax + b to data:

    >>> x_data = torch.tensor([0., 1., 2., 3.])
    >>> y_data = torch.tensor([1., 3., 5., 7.])  # y = 2x + 1
    >>> def residuals(params):
    ...     a, b = params[0], params[1]
    ...     return a * x_data + b - y_data
    >>> result = levenberg_marquardt(residuals, torch.zeros(2))
    >>> result
    tensor([2., 1.])

    The optimizer supports implicit differentiation:

    >>> target = torch.tensor([5.0], requires_grad=True)
    >>> def residuals(x):
    ...     return x - target
    >>> result = levenberg_marquardt(residuals, torch.zeros(1))
    >>> result.backward()
    >>> target.grad
    tensor([1.])

    References
    ----------
    - Levenberg, K. "A method for the solution of certain non-linear
      problems in least squares." Quarterly of applied mathematics 2.2
      (1944): 164-168.
    - Marquardt, D.W. "An algorithm for least-squares estimation of
      nonlinear parameters." Journal of the society for Industrial and
      Applied Mathematics 11.2 (1963): 431-441.

    See Also
    --------
    https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
    """
    if tol is None:
        tol = torch.finfo(x0.dtype).eps ** 0.5

    x = x0.clone()
    mu = damping
    n = x.numel()

    for _ in range(maxiter):
        r = residuals(x)

        if jacobian is not None:
            J = jacobian(x)
        else:
            J = torch.func.jacrev(residuals)(x)

        # Ensure J is 2D
        if J.dim() == 1:
            J = J.unsqueeze(0)

        # Gradient: g = J^T @ r
        g = J.T @ r

        # Check convergence
        if torch.norm(g) < tol:
            break

        # Hessian approximation: H = J^T @ J + mu * I
        JtJ = J.T @ J
        H = JtJ + mu * torch.eye(n, dtype=x.dtype, device=x.device)

        # Solve H @ delta = -g
        try:
            delta = torch.linalg.solve(H, -g)
        except RuntimeError:
            # Matrix is singular, increase damping
            mu *= 10
            continue

        # Compute actual vs predicted reduction
        x_new = x + delta
        r_new = residuals(x_new)

        actual_reduction = torch.sum(r**2) - torch.sum(r_new**2)
        predicted_reduction = -2 * (g @ delta) - delta @ JtJ @ delta

        # Avoid division by zero
        rho = actual_reduction / (predicted_reduction + 1e-10)

        if rho > 0.25:
            # Good step, accept and decrease damping
            x = x_new
            mu = max(mu / 3, 1e-10)
        else:
            # Bad step, reject and increase damping
            mu = min(mu * 2, 1e10)

    # Attach implicit gradient for backpropagation
    return _LMImplicitGrad.apply(x, residuals)
