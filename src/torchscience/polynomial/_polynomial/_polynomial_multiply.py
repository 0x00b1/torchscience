import torch

from ._polynomial import Polynomial


def polynomial_multiply(p: Polynomial, q: Polynomial) -> Polynomial:
    """Multiply two polynomials.

    Computes convolution of coefficients. Result degree is deg(p) + deg(q).

    Parameters
    ----------
    p, q : Polynomial
        Polynomials to multiply.

    Returns
    -------
    Polynomial
        Product p * q.
    """
    p_coeffs = p.coeffs
    q_coeffs = q.coeffs

    # Get shapes
    p_batch = p_coeffs.shape[:-1]
    q_batch = q_coeffs.shape[:-1]
    n_p = p_coeffs.shape[-1]
    n_q = q_coeffs.shape[-1]

    # Handle empty polynomials
    if n_p == 0 or n_q == 0:
        broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)
        return Polynomial(
            coeffs=torch.zeros(
                *broadcast_batch,
                0,
                dtype=p_coeffs.dtype,
                device=p_coeffs.device,
            )
        )

    # Broadcast batch dimensions
    broadcast_batch = torch.broadcast_shapes(p_batch, q_batch)

    # Expand to broadcast shape
    p_expanded = p_coeffs.expand(*broadcast_batch, n_p)
    q_expanded = q_coeffs.expand(*broadcast_batch, n_q)

    # Flatten batch dimensions for kernel: (...batch, N) -> (B, N)
    batch_size = broadcast_batch.numel() if len(broadcast_batch) > 0 else 1
    p_flat = p_expanded.reshape(batch_size, n_p).contiguous()
    q_flat = q_expanded.reshape(batch_size, n_q).contiguous()

    # Promote to common dtype
    common_dtype = torch.promote_types(p_flat.dtype, q_flat.dtype)
    p_flat = p_flat.to(common_dtype)
    q_flat = q_flat.to(common_dtype)

    # Call C++ kernel
    result_flat = torch.ops.torchscience.polynomial_multiply(p_flat, q_flat)

    # Reshape output back to broadcast batch dimensions
    n_out = n_p + n_q - 1
    if len(broadcast_batch) == 0:
        result = result_flat.reshape(n_out)
    else:
        result = result_flat.reshape(*broadcast_batch, n_out)

    return Polynomial(coeffs=result)
