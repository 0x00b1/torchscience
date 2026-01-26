"""Nonlinear curve fitting via optimization."""

from typing import Callable, Optional

import torch
from torch import Tensor

from torchscience.optimization._result import OptimizeResult
from torchscience.optimization.minimization._l_bfgs import l_bfgs
from torchscience.optimization.minimization._levenberg_marquardt import (
    levenberg_marquardt,
)


def curve_fit(
    f: Callable[[Tensor, Tensor], Tensor],
    xdata: Tensor,
    ydata: Tensor,
    p0: Tensor,
    *,
    sigma: Optional[Tensor] = None,
    method: str = "levenberg-marquardt",
    **kwargs,
) -> OptimizeResult:
    r"""Fit a function to data using nonlinear least squares.

    Finds parameters ``p`` that minimize:

    .. math::

        \min_p \sum_i \left(\frac{f(x_i, p) - y_i}{\sigma_i}\right)^2

    Parameters
    ----------
    f : Callable[[Tensor, Tensor], Tensor]
        Model function ``f(xdata, params) -> ydata``. Takes data points
        and parameters, returns predicted values.
    xdata : Tensor
        Independent variable data.
    ydata : Tensor
        Dependent variable data to fit.
    p0 : Tensor
        Initial parameter guess of shape ``(n,)``.
    sigma : Tensor, optional
        Standard deviations of ``ydata``. If provided, the residuals are
        weighted by ``1 / sigma``. Must have the same shape as ``ydata``.
    method : str
        Optimization method: ``"levenberg-marquardt"`` (default) or
        ``"l-bfgs"``.
    **kwargs
        Additional keyword arguments passed to the underlying solver.

    Returns
    -------
    OptimizeResult
        Optimization result with the fitted parameters in ``x``.

    Examples
    --------
    Fit a line ``y = a*x + b``:

    >>> import torch
    >>> xdata = torch.tensor([0., 1., 2., 3.])
    >>> ydata = torch.tensor([1., 3., 5., 7.])  # y = 2x + 1
    >>> def model(x, params):
    ...     return params[0] * x + params[1]
    >>> result = curve_fit(model, xdata, ydata, torch.zeros(2))
    >>> result.x
    tensor([2., 1.])

    Fit an exponential ``y = a * exp(-b * x)``:

    >>> xdata = torch.linspace(0, 4, 10)
    >>> ydata = 2.0 * torch.exp(-0.5 * xdata)
    >>> def model(x, params):
    ...     return params[0] * torch.exp(-params[1] * x)
    >>> result = curve_fit(model, xdata, ydata, torch.tensor([1.0, 1.0]))

    Weighted fit with known uncertainties:

    >>> sigma = torch.ones(4) * 0.1
    >>> result = curve_fit(model, xdata[:4], ydata[:4], torch.tensor([1.0, 1.0]), sigma=sigma)

    See Also
    --------
    levenberg_marquardt : Levenberg-Marquardt solver.
    l_bfgs : L-BFGS solver.
    """
    method_lower = method.lower().replace("_", "-")

    def residuals(params: Tensor) -> Tensor:
        r = f(xdata, params) - ydata
        if sigma is not None:
            r = r / sigma
        return r

    if method_lower == "levenberg-marquardt":
        lm_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("jacobian", "tol", "maxiter", "damping")
        }
        x_opt = levenberg_marquardt(residuals, p0, **lm_kwargs)

        with torch.no_grad():
            r_final = residuals(x_opt)
            f_val = (r_final**2).sum()
            tol = lm_kwargs.get("tol")
            if tol is None:
                tol = torch.finfo(p0.dtype).eps ** 0.5
            grad_norm = torch.norm(r_final)
            converged = torch.tensor(
                grad_norm.item() < tol,
                device=p0.device,
            )

        return OptimizeResult(
            x=x_opt,
            converged=converged,
            num_iterations=torch.tensor(
                lm_kwargs.get("maxiter", 100),
                dtype=torch.int64,
                device=p0.device,
            ),
            fun=f_val,
        )

    if method_lower == "l-bfgs":

        def objective(params: Tensor) -> Tensor:
            r = residuals(params)
            return (r**2).sum()

        lbfgs_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ("grad", "maxiter", "tol", "history_size", "line_search")
        }
        return l_bfgs(objective, p0, **lbfgs_kwargs)

    raise ValueError(
        f"Unknown method {method!r}. Supported methods: "
        f"'levenberg-marquardt', 'l-bfgs'."
    )
