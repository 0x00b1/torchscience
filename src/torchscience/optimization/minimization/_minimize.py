"""Unified minimize interface for unconstrained optimization."""

from typing import Callable, Optional

from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import l_bfgs
from torchscience.optimization.minimization._levenberg_marquardt import (
    levenberg_marquardt,
)


def minimize(
    fun: Callable[[Tensor], Tensor],
    x0: Tensor,
    *,
    method: str = "l-bfgs",
    grad: Optional[Callable[[Tensor], Tensor]] = None,
    **kwargs,
) -> OptimizeResult:
    r"""Unified interface for unconstrained minimization.

    Dispatches to the specified solver to find:

    .. math::

        \min_x f(x)

    Parameters
    ----------
    fun : Callable[[Tensor], Tensor]
        Objective function to minimize.
    x0 : Tensor
        Initial guess.
    method : str
        Optimization method: ``"l-bfgs"`` (default) or
        ``"levenberg-marquardt"``.
    grad : Callable, optional
        Gradient function. Only used by ``"l-bfgs"``.
    **kwargs
        Additional keyword arguments passed to the solver.

    Returns
    -------
    OptimizeResult
        Optimization result with fields ``x``, ``converged``,
        ``num_iterations``, and ``fun``.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.

    Examples
    --------
    >>> import torch
    >>> def f(x):
    ...     return (x ** 2).sum()
    >>> result = minimize(f, torch.tensor([3.0, 4.0]))
    >>> result.x
    tensor([0., 0.])

    >>> result = minimize(f, torch.tensor([3.0]), method="l-bfgs")
    >>> result.converged
    tensor(True)

    See Also
    --------
    l_bfgs : L-BFGS solver.
    levenberg_marquardt : Levenberg-Marquardt solver for least squares.
    """
    method_lower = method.lower().replace("_", "-")

    if method_lower == "l-bfgs":
        return l_bfgs(fun, x0, grad=grad, **kwargs)

    if method_lower == "levenberg-marquardt":
        # LM expects a residuals function and returns a Tensor.
        # Wrap its result in OptimizeResult.
        import torch

        lm_kwargs = {k: v for k, v in kwargs.items() if k != "grad"}
        if "jacobian" not in lm_kwargs and grad is not None:
            lm_kwargs["jacobian"] = grad
        x_opt = levenberg_marquardt(fun, x0, **lm_kwargs)

        with torch.no_grad():
            f_val = fun(x_opt)
            # For LM, convergence is not tracked; report based on result
            residual_norm = (
                f_val.abs().sum() if f_val.dim() > 0 else f_val.abs()
            )

        tol = kwargs.get("tol")
        if tol is None:
            tol = torch.finfo(x0.dtype).eps ** 0.5
        converged = torch.tensor(
            residual_norm.item() < tol,
            device=x0.device,
        )
        return OptimizeResult(
            x=x_opt,
            converged=converged,
            num_iterations=torch.tensor(
                kwargs.get("maxiter", 100),
                dtype=torch.int64,
                device=x0.device,
            ),
            fun=f_val,
        )

    raise ValueError(
        f"Unknown method {method!r}. Supported methods: 'l-bfgs', "
        f"'levenberg-marquardt'."
    )
