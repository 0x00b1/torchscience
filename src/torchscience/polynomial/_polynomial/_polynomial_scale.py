from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._polynomial import Polynomial


def polynomial_scale(p: "Polynomial", c: Tensor) -> "Polynomial":
    """Multiply polynomial by scalar(s).

    Uses the polynomial_scale C++ kernel with full autograd support for
    computing gradients with respect to both the polynomial coefficients
    and the scalar.

    Parameters
    ----------
    p : Polynomial
        Polynomial to scale.
    c : Tensor
        Scalar(s), broadcasts with batch dimensions.

    Returns
    -------
    Polynomial
        Scaled polynomial c * p.
    """
    # p IS the coefficient tensor now
    # Get shapes
    p_batch = p.shape[:-1]
    n_coeffs = p.shape[-1]

    # Handle scalar vs batched c
    if c.dim() == 0:
        # Scalar c - expand to match batch size
        c_expanded = c
        broadcast_batch = p_batch
    else:
        # c has batch dimensions - broadcast with p
        c_batch = c.shape
        broadcast_batch = torch.broadcast_shapes(p_batch, c_batch)
        c_expanded = c.expand(*broadcast_batch)

    # Expand p to broadcast shape
    p_expanded = p.expand(*broadcast_batch, n_coeffs)

    # Flatten batch dimensions for kernel: (...batch, N) -> (B, N)
    batch_size = broadcast_batch.numel() if len(broadcast_batch) > 0 else 1
    p_flat = p_expanded.reshape(batch_size, n_coeffs).contiguous()

    # c needs to be (B,) for the kernel
    if c.dim() == 0:
        c_flat = c_expanded.expand(batch_size)
    else:
        c_flat = c_expanded.reshape(batch_size).contiguous()

    # Promote to common dtype
    common_dtype = torch.promote_types(p_flat.dtype, c_flat.dtype)
    p_flat = p_flat.to(common_dtype)
    c_flat = c_flat.to(common_dtype)

    # Call C++ kernel
    result_flat = torch.ops.torchscience.polynomial_scale(p_flat, c_flat)

    # Reshape output back to broadcast batch dimensions
    if len(broadcast_batch) == 0:
        result = result_flat.reshape(n_coeffs)
    else:
        result = result_flat.reshape(*broadcast_batch, n_coeffs)

    from ._polynomial import polynomial

    return polynomial(result)
