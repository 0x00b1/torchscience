"""Convergence utilities for root finding."""

import torch
from torch import Tensor


def default_tolerances(dtype: torch.dtype) -> dict[str, float]:
    """Return dtype-appropriate default tolerances.

    Parameters
    ----------
    dtype : torch.dtype
        The tensor dtype.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'xtol', 'rtol', 'ftol'.
    """
    if dtype in (torch.float16, torch.bfloat16):
        return {"xtol": 1e-3, "rtol": 1e-2, "ftol": 1e-3}
    elif dtype == torch.float32:
        return {"xtol": 1e-6, "rtol": 1e-5, "ftol": 1e-6}
    else:  # float64 and others
        return {"xtol": 1e-12, "rtol": 1e-9, "ftol": 1e-12}


def check_convergence(
    x_old: Tensor,
    x_new: Tensor,
    f_new: Tensor,
    xtol: float,
    rtol: float,
    ftol: float,
) -> Tensor:
    """Check convergence for each element.

    Convergence is achieved when EITHER:
    - |x_new - x_old| < xtol + rtol * |x_old| (x converged)
    - |f_new| < ftol (f converged)

    Parameters
    ----------
    x_old : Tensor
        Previous x values.
    x_new : Tensor
        Current x values.
    f_new : Tensor
        Current function values.
    xtol : float
        Absolute tolerance on x.
    rtol : float
        Relative tolerance on x.
    ftol : float
        Tolerance on function value.

    Returns
    -------
    Tensor
        Boolean mask where True indicates convergence.
    """
    x_converged = torch.abs(x_new - x_old) < xtol + rtol * torch.abs(x_old)
    f_converged = torch.abs(f_new) < ftol
    return x_converged | f_converged
