"""Yule-Walker IIR filter design from autocorrelation sequence."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def yule_walker(
    r: Tensor,
    order: int,
    allow_singular: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, Tensor]:
    """
    Solve the Yule-Walker equations for autoregressive (AR) model parameters.

    Parameters
    ----------
    r : Tensor
        Autocorrelation sequence, shape (order+1,) or larger.
        r[k] is the autocorrelation at lag k.
    order : int
        Order of the AR model (number of AR coefficients).
    allow_singular : bool, optional
        If True, handle singular matrices gracefully. Default is False.
    dtype : torch.dtype, optional
        Output dtype. If None, uses dtype of r.
    device : torch.device, optional
        Output device. If None, uses device of r.

    Returns
    -------
    a : Tensor
        AR coefficients, shape (order,).
    sigma : Tensor
        Standard deviation of the prediction error (scalar).

    Notes
    -----
    Uses Levinson-Durbin recursion for O(n^2) efficiency.
    """
    if dtype is None:
        dtype = r.dtype
    if device is None:
        device = r.device

    r = r.to(dtype=dtype, device=device)

    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    if order >= len(r):
        raise ValueError(
            f"order must be less than length of r, got order={order}, len(r)={len(r)}"
        )

    # Levinson-Durbin recursion
    a = torch.zeros(order, dtype=dtype, device=device)
    e = r[0]

    if e == 0:
        if allow_singular:
            return a, torch.tensor(0.0, dtype=dtype, device=device)
        else:
            raise ValueError(
                "r[0] is zero, autocorrelation matrix is singular"
            )

    for i in range(order):
        # Compute reflection coefficient
        if i == 0:
            lam = r[1] / e
        else:
            # r[1:i+1] reversed is r[i], r[i-1], ..., r[1]
            sum_term = torch.sum(a[:i] * r[1 : i + 1].flip(0))
            lam = (r[i + 1] - sum_term) / e

        # Update coefficients
        if i > 0:
            a_prev = a[:i].clone()
            a[:i] = a_prev - lam * a_prev.flip(0)
        a[i] = lam

        # Update prediction error
        e = e * (1 - lam * lam)

        if e <= 0:
            if allow_singular:
                e = torch.tensor(0.0, dtype=dtype, device=device)
                break
            else:
                raise ValueError("Prediction error became non-positive")

    sigma = (
        torch.sqrt(e)
        if isinstance(e, Tensor)
        else torch.tensor(e, dtype=dtype, device=device).sqrt()
    )

    return a, sigma
