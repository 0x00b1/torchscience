from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ._polynomial import Polynomial


def polynomial_subtract(p: "Polynomial", q: "Polynomial") -> "Polynomial":
    """Subtract two polynomials.

    Computes element-wise difference of coefficients with zero-padding for
    polynomials of different degrees.

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to subtract.

    Returns
    -------
    Polynomial
        Difference p - q.
    """
    # p and q ARE the coefficient tensors now
    # Get shapes
    p_batch = p.shape[:-1]
    q_batch = q.shape[:-1]
    n_p = p.shape[-1]
    n_q = q.shape[-1]

    # Broadcast batch dimensions
    broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)

    # Expand to broadcast shape
    p_expanded = p.expand(*broadcast_batch, n_p)
    q_expanded = q.expand(*broadcast_batch, n_q)

    # Flatten batch dimensions for kernel: (...batch, N) -> (B, N)
    batch_size = broadcast_batch.numel() if len(broadcast_batch) > 0 else 1
    p_flat = p_expanded.reshape(batch_size, n_p).contiguous()
    q_flat = q_expanded.reshape(batch_size, n_q).contiguous()

    # Promote to common dtype
    common_dtype = torch.promote_types(p_flat.dtype, q_flat.dtype)
    p_flat = p_flat.to(common_dtype)
    q_flat = q_flat.to(common_dtype)

    # Call C++ kernel
    result_flat = torch.ops.torchscience.polynomial_subtract(p_flat, q_flat)

    # Reshape output back to broadcast batch dimensions
    n_out = max(n_p, n_q)
    if len(broadcast_batch) == 0:
        result = result_flat.reshape(n_out)
    else:
        result = result_flat.reshape(*broadcast_batch, n_out)

    from ._polynomial import polynomial

    return polynomial(result)
