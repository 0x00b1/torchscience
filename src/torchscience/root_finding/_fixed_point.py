"""Fixed-point iteration method."""

from typing import Callable

import torch
from torch import Tensor

from ._convergence import default_tolerances


def _check_fixed_point_convergence(
    x_old: Tensor,
    x_new: Tensor,
    xtol: float,
    rtol: float,
    is_system: bool,
) -> Tensor:
    """Check convergence for fixed-point iteration.

    Unlike root-finding methods, fixed-point iteration only checks x change,
    not function residual.

    Parameters
    ----------
    x_old : Tensor
        Previous x values.
    x_new : Tensor
        Current x values.
    xtol : float
        Absolute tolerance on x.
    rtol : float
        Relative tolerance on x.
    is_system : bool
        If True, x has shape (B, n) and we check max change across system dimension.
        If False, x has shape (B,) and we check element-wise.

    Returns
    -------
    Tensor
        Boolean mask where True indicates convergence. Shape (B,).
    """
    diff = torch.abs(x_new - x_old)
    tol = xtol + rtol * torch.abs(x_old)

    if is_system:
        # For systems, check that max change across system dimension is within tolerance
        # diff has shape (B, n), we want to check if all elements in each row converged
        return (diff < tol).all(dim=-1)
    else:
        # For scalar, element-wise convergence
        return diff < tol


def fixed_point(
    g: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    xtol: float | None = None,
    rtol: float | None = None,
    maxiter: int = 500,
) -> tuple[Tensor, Tensor]:
    """
    Find fixed point of g(x) = x using simple iteration.

    The fixed-point iteration repeatedly applies the function g until
    convergence: x_{n+1} = g(x_n). This converges when g is a contraction
    mapping, i.e., |g(x) - g(y)| < |x - y| for all x, y in the domain.

    Parameters
    ----------
    g : Callable[[Tensor], Tensor]
        Vectorized function that maps x to g(x). The fixed point satisfies
        g(x*) = x*. For scalar problems, takes tensor of shape ``(N,)`` and
        returns ``(N,)``. For systems, takes tensor of shape ``(B, n)`` and
        returns ``(B, n)``.
    x0 : Tensor
        Initial guess for the fixed point. Shape ``(N,)`` for scalar problems
        or ``(B, n)`` for systems where B is batch size and n is system dimension.
    xtol : float, optional
        Absolute tolerance on x change. Convergence requires
        ``|x_new - x_old| < xtol + rtol * |x_old|``.
        Default: dtype-aware (1e-3 for float16/bfloat16, 1e-6 for float32,
        1e-12 for float64).
    rtol : float, optional
        Relative tolerance on x change. Combined with xtol for convergence.
        Default: dtype-aware (1e-2 for float16/bfloat16, 1e-5 for float32,
        1e-9 for float64).
    maxiter : int, default=500
        Maximum iterations. Non-converged elements will have converged=False.

    Returns
    -------
    tuple[Tensor, Tensor]
        - **fixed_point** -- Fixed points with the same shape as input ``x0``.
          For non-converged elements, this is the best estimate.
        - **converged** -- Boolean tensor indicating which elements converged
          within maxiter iterations. Shape ``(N,)`` for scalar problems or
          ``(B,)`` for systems.

    Examples
    --------
    Find the fixed point of cos(x) (Dottie number):

    >>> import torch
    >>> from torchscience.root_finding import fixed_point
    >>> g = lambda x: torch.cos(x)
    >>> x0 = torch.tensor([1.0])
    >>> fp, converged = fixed_point(g, x0)
    >>> float(fp)  # doctest: +ELLIPSIS
    0.739...
    >>> converged.all()
    tensor(True)

    Find sqrt(2) using the iteration x = (x + 2/x) / 2:

    >>> g = lambda x: (x + 2/x) / 2
    >>> x0 = torch.tensor([1.5])
    >>> fp, converged = fixed_point(g, x0)
    >>> float(fp)  # doctest: +ELLIPSIS
    1.414...

    Batched fixed-point iteration:

    >>> c = torch.tensor([2.0, 3.0, 4.0])
    >>> g = lambda x: (x + c/x) / 2  # Babylonian method for sqrt(c)
    >>> x0 = torch.tensor([1.5, 1.5, 1.5])
    >>> fps, converged = fixed_point(g, x0)
    >>> [f"{v:.4f}" for v in fps.tolist()]
    ['1.4142', '1.7321', '2.0000']

    Fixed-point iteration for a 2D system:

    >>> def g(x):
    ...     # x has shape (B, 2)
    ...     x1, x2 = x[..., 0], x[..., 1]
    ...     return torch.stack([0.5 * x2, 0.5 * x1 + 0.1], dim=-1)
    >>> x0 = torch.tensor([[1.0, 1.0]])  # Shape (1, 2)
    >>> fp, converged = fixed_point(g, x0)
    >>> fp.shape
    torch.Size([1, 2])

    Notes
    -----
    **Convergence**: Fixed-point iteration converges when g is a contraction
    mapping. The rate of convergence depends on the Lipschitz constant of g.
    For g with Lipschitz constant L < 1, the method converges linearly with
    rate L.

    **No ftol**: Unlike root-finding methods that check |f(x)| < ftol,
    fixed-point iteration only checks the change in x. This is because
    the "residual" g(x) - x should equal zero at convergence, which is
    equivalent to checking the x change.

    **Systems**: For system inputs with shape (B, n), convergence is checked
    by requiring that all components of x satisfy the tolerance condition.
    The returned converged tensor has shape (B,).

    **CUDA Support**: Works on any device (CPU or CUDA) as long as all
    inputs are on the same device.

    See Also
    --------
    newton : Newton-Raphson method for root-finding
    scipy.optimize.fixed_point : SciPy's fixed-point iteration
    """
    # Determine if this is a system (2D input) or scalar (1D input)
    orig_shape = x0.shape
    is_system = x0.dim() >= 2

    if x0.numel() == 0:
        if is_system:
            batch_shape = orig_shape[:-1]
            empty_converged = torch.ones(
                batch_shape, dtype=torch.bool, device=x0.device
            )
        else:
            empty_converged = torch.ones(
                orig_shape, dtype=torch.bool, device=x0.device
            )
        return x0.clone(), empty_converged

    x = x0.clone()

    # Get tolerances
    dtype = x.dtype
    defaults = default_tolerances(dtype)
    if xtol is None:
        xtol = defaults["xtol"]
    if rtol is None:
        rtol = defaults["rtol"]

    # Track which elements have converged
    if is_system:
        # For systems, converged has shape matching batch dimensions
        batch_shape = orig_shape[:-1]
        converged = torch.zeros(batch_shape, dtype=torch.bool, device=x.device)
    else:
        converged = torch.zeros(orig_shape, dtype=torch.bool, device=x.device)

    result = x.clone()

    for _ in range(maxiter):
        # Fixed-point iteration: x_new = g(x)
        x_new = g(x)

        # Check convergence
        newly_converged = _check_fixed_point_convergence(
            x, x_new, xtol, rtol, is_system
        )
        newly_converged = newly_converged & ~converged

        # Update results for newly converged elements
        if is_system:
            # Expand newly_converged to match x shape for indexing
            mask = newly_converged.unsqueeze(-1).expand_as(x)
            result = torch.where(mask, x_new, result)
        else:
            result = torch.where(newly_converged, x_new, result)

        converged = converged | newly_converged

        if torch.all(converged):
            return result, converged

        # Update x for next iteration (only unconverged elements)
        if is_system:
            mask = converged.unsqueeze(-1).expand_as(x)
            x = torch.where(mask, x, x_new)
        else:
            x = torch.where(converged, x, x_new)

    # Return best estimate for non-converged elements
    if is_system:
        mask = converged.unsqueeze(-1).expand_as(x)
        result = torch.where(mask, result, x)
    else:
        result = torch.where(converged, result, x)

    return result, converged
