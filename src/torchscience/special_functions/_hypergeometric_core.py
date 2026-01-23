from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor


def _as_param_tensor(x: Optional[Iterable[Tensor | float] | Tensor],
                     dtype: torch.dtype,
                     device: torch.device,
                     batch_shape: Tuple[int, ...]) -> Tuple[Tensor, int]:
    if x is None:
        t = torch.ones((*batch_shape, 0), dtype=dtype, device=device)
        return t, 0
    if isinstance(x, Tensor):
        t = x
    else:
        # Convert iterable of scalars/Tensors into a stacked param-dim
        elems = []
        for e in list(x):
            if isinstance(e, Tensor):
                e_t = e.to(dtype=dtype, device=device)
            else:
                e_t = torch.as_tensor(e, dtype=dtype, device=device)
            # Broadcast element to batch shape
            e_t = e_t.expand(batch_shape)
            elems.append(e_t)
        if len(elems) == 0:
            t = torch.ones((*batch_shape, 0), dtype=dtype, device=device)
            return t, 0
        t = torch.stack(elems, dim=-1)
        if t.ndim == 0:
            t = t.unsqueeze(0)
    p = t.shape[-1]
    # Broadcast to batch shape
    if t.shape[:-1] != batch_shape:
        t = t.expand((*batch_shape, p))
    return t, p


def _promote_dtype(z: Tensor) -> torch.dtype:
    if z.is_complex():
        # Ensure at least complex64
        return torch.complex64 if z.dtype == torch.complex64 else torch.complex128
    # Promote low-precision to float32
    if z.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return z.dtype


def _pfq_series(
    a_params: Optional[Iterable[Tensor | float] | Tensor],
    b_params: Optional[Iterable[Tensor | float] | Tensor],
    z: Tensor,
    *,
    tol: float = 1e-12,
    max_terms: int = 2048,
    min_terms: int = 8,
    accel: bool = False,  # reserved, not used initially
    _debug: bool = False,
) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate the generalized hypergeometric function pFq via its series.

    Parameters
    ----------
    a_params : sequence or Tensor or None
        Sequence of `p` parameters a_i (p can be 0). Broadcasts over batch dims.
    b_params : sequence or Tensor or None
        Sequence of `q` parameters b_j (q can be 0). Broadcasts over batch dims.
    z : Tensor
        Argument tensor. Broadcasts with parameters.
    tol : float
        Relative tolerance for adaptive truncation.
    max_terms : int
        Maximum number of series terms.
    min_terms : int
        Minimum number of terms before allowing early stopping.
    accel : bool
        Placeholder for future series acceleration (e.g., Wynn epsilon).
    _debug : bool
        If True, also return (terms_used, converged_mask).
    """
    if not isinstance(z, Tensor):
        raise TypeError("z must be a torch.Tensor")

    # Compute broadcast batch shape across inputs (excluding parameter dims)
    a = a_params if isinstance(a_params, Tensor) else None
    b = b_params if isinstance(b_params, Tensor) else None
    a_bs = () if a is None else a.shape[:-1] if a.ndim > 0 else ()
    b_bs = () if b is None else b.shape[:-1] if b.ndim > 0 else ()
    z_bs = z.shape
    batch_shape = torch.broadcast_shapes(a_bs, b_bs, z_bs)

    dtype = _promote_dtype(z)
    device = z.device

    z_b = z.to(dtype).expand(batch_shape)
    a_b, p = _as_param_tensor(a_params, dtype, device, batch_shape)
    b_b, q = _as_param_tensor(b_params, dtype, device, batch_shape)

    # Initialize series
    S = torch.ones(batch_shape, dtype=dtype, device=device)
    t = S.clone()
    converged = torch.zeros(batch_shape, dtype=torch.bool, device=device)
    terms_used = torch.zeros(batch_shape, dtype=torch.int32, device=device)

    # Helper to compute product over last dim safely when size==0 → 1
    def prod_last(x: Tensor) -> Tensor:
        if x.shape[-1] == 0:
            return torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
        return x.prod(dim=-1)

    # Iteratively accumulate terms t_k
    k_scalar = torch.tensor(0.0, dtype=dtype, device=device)
    for k in range(1, max_terms + 1):
        # Build factors (a_i + k-1) and (b_j + k-1)
        k_minus_1 = k_scalar
        if p:
            num = prod_last(a_b + k_minus_1)
        else:
            num = torch.ones_like(S)
        if q:
            den_core = prod_last(b_b + k_minus_1)
        else:
            den_core = torch.ones_like(S)

        # Denominator includes k
        den = den_core * torch.as_tensor(float(k), dtype=dtype, device=device)

        # Guard division by zero: where den==0, set ratio to +inf (propagate to S)
        safe_den = torch.where(den == 0, torch.ones_like(den), den)
        ratio = (num / safe_den) * z_b
        ratio = torch.where(den == 0, torch.full_like(ratio, float("inf")), ratio)

        t = t * ratio
        S = S + t

        # Track terms used (first index when converged)
        just_converged = (~converged) & (t.abs() <= (S.abs() * tol)) & (k >= min_terms)
        terms_used = torch.where(just_converged, torch.as_tensor(k, dtype=torch.int32, device=device), terms_used)
        converged = converged | just_converged

        # Early exit if all converged
        if bool(converged.all()):
            break

        # update k scalar
        k_scalar = k_scalar + 1.0

    if _debug:
        return S, terms_used, converged
    return S


__all__ = ["_pfq_series"]
