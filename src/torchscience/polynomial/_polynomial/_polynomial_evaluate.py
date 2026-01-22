import torch
from torch import Tensor

from ._polynomial import Polynomial


def polynomial_evaluate(p: Polynomial, x: Tensor) -> Tensor:
    """Evaluate polynomial at points using Horner's method.

    Parameters
    ----------
    p : Polynomial
        Polynomial with coefficients shape (...batch, N).
    x : Tensor
        Evaluation points, shape (...x_batch). The result shape is
        (...batch, ...x_batch) where batch dimensions of p broadcast
        with x.

    Returns
    -------
    Tensor
        Values p(x), shape is broadcast of p's batch dims with x's shape.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
    >>> polynomial_evaluate(p, torch.tensor([0.0, 1.0, 2.0]))
    tensor([ 1.,  6., 17.])
    """
    # p IS the coefficient tensor now
    if p.shape[-1] == 0:
        return x * 0.0

    batch_shape = p.shape[:-1]
    x_shape = x.shape
    N = p.shape[-1]

    # Flatten batch dimensions: (...batch, N) -> (B, N)
    B = p[..., 0].numel() if len(batch_shape) > 0 else 1
    M = x.numel()

    # Clone to avoid issues with in-place modifications after evaluate
    coeffs_flat = p.reshape(B, N).contiguous().clone()
    x_flat = x.reshape(M).contiguous().clone()

    # Promote to common dtype (x must match coeffs dtype for the kernel)
    common_dtype = torch.promote_types(coeffs_flat.dtype, x_flat.dtype)
    coeffs_flat = coeffs_flat.to(common_dtype)
    x_flat = x_flat.to(common_dtype)

    # Call C++ kernel: (B, N) x (M,) -> (B, M)
    result_flat = torch.ops.torchscience.polynomial_evaluate(
        coeffs_flat, x_flat
    )

    # Reshape output to (...batch, ...x_shape)
    output_shape = batch_shape + x_shape
    return result_flat.reshape(output_shape)
