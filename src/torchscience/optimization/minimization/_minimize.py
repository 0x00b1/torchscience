"""Unified minimize interface for unconstrained optimization."""

from typing import Callable, Optional

from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._conjugate_gradient import (
    conjugate_gradient,
)
from torchscience.optimization.minimization._l_bfgs import l_bfgs
from torchscience.optimization.minimization._levenberg_marquardt import (
    levenberg_marquardt,
)
from torchscience.optimization.minimization._nelder_mead import nelder_mead
from torchscience.optimization.minimization._newton_cg import newton_cg
from torchscience.optimization.minimization._trust_region import trust_region


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
        Optimization method: ``"l-bfgs"`` (default),
        ``"conjugate-gradient"``, ``"newton-cg"``, ``"trust-region"``,
        ``"nelder-mead"``, or ``"levenberg-marquardt"``.
    grad : Callable, optional
        Gradient function. Used by gradient-based methods
        (``"l-bfgs"``, ``"conjugate-gradient"``, ``"newton-cg"``,
        ``"trust-region"``).
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
    conjugate_gradient : Conjugate gradient solver.
    newton_cg : Newton-CG solver.
    trust_region : Trust-region solver.
    nelder_mead : Nelder-Mead simplex solver.
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

    if method_lower == "conjugate-gradient":
        cg_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("maxiter", "tol", "variant", "line_search")
        }
        return conjugate_gradient(fun, x0, grad=grad, **cg_kwargs)

    if method_lower == "newton-cg":
        ncg_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("maxiter", "tol", "max_cg_iter", "line_search")
        }
        return newton_cg(fun, x0, grad=grad, **ncg_kwargs)

    if method_lower == "trust-region":
        tr_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "maxiter",
                "tol",
                "max_cg_iter",
                "initial_trust_radius",
                "max_trust_radius",
                "eta",
            )
        }
        return trust_region(fun, x0, grad=grad, **tr_kwargs)

    if method_lower == "nelder-mead":
        nm_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "maxiter",
                "tol",
                "initial_simplex",
                "alpha",
                "gamma",
                "rho",
                "sigma",
            )
        }
        return nelder_mead(fun, x0, **nm_kwargs)

    raise ValueError(
        f"Unknown method {method!r}. Supported methods: 'l-bfgs', "
        f"'conjugate-gradient', 'newton-cg', 'trust-region', "
        f"'nelder-mead', 'levenberg-marquardt'."
    )
