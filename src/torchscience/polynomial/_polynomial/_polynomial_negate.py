import torch

from ._polynomial import Polynomial


def polynomial_negate(p: Polynomial) -> Polynomial:
    """Negate polynomial.

    Computes element-wise negation of coefficients.

    Parameters
    ----------
    p : Polynomial
        Polynomial to negate.

    Returns
    -------
    Polynomial
        Negated polynomial -p.
    """
    p_coeffs = p.coeffs

    # Get shapes
    p_batch = p_coeffs.shape[:-1]
    n_p = p_coeffs.shape[-1]

    # Flatten batch dimensions for kernel: (...batch, N) -> (B, N)
    batch_size = p_batch.numel() if len(p_batch) > 0 else 1
    p_flat = p_coeffs.reshape(batch_size, n_p).contiguous()

    # Call C++ kernel
    result_flat = torch.ops.torchscience.polynomial_negate(p_flat)

    # Reshape output back to original batch dimensions
    if len(p_batch) == 0:
        result = result_flat.reshape(n_p)
    else:
        result = result_flat.reshape(*p_batch, n_p)

    return Polynomial(coeffs=result)
