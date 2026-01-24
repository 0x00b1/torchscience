from typing import TYPE_CHECKING

import torch

from torchscience.polynomial._degree_error import DegreeError

if TYPE_CHECKING:
    from ._polynomial import Polynomial


def polynomial_divmod(
    p: "Polynomial", q: "Polynomial"
) -> tuple["Polynomial", "Polynomial"]:
    """Divide polynomial p by q, returning quotient and remainder.

    Computes quotient and remainder such that p = q * quotient + remainder,
    where deg(remainder) < deg(q).

    Parameters
    ----------
    p : Polynomial
        Dividend polynomial.
    q : Polynomial
        Divisor polynomial. Leading coefficient must be non-zero.

    Returns
    -------
    quotient : Polynomial
        Quotient of division.
    remainder : Polynomial
        Remainder of division.

    Raises
    ------
    DegreeError
        If divisor is zero polynomial or dividend degree < divisor degree.

    Examples
    --------
    >>> p = polynomial(torch.tensor([-1.0, 0.0, 0.0, 1.0]))  # x^3 - 1
    >>> q = polynomial(torch.tensor([-1.0, 1.0]))  # x - 1
    >>> quot, rem = polynomial_divmod(p, q)
    >>> quot  # x^2 + x + 1
    Polynomial(tensor([1., 1., 1.]))
    """
    # p and q ARE the coefficient tensors now
    # Get degrees
    n_p = p.shape[-1]
    n_q = q.shape[-1]
    deg_p = n_p - 1
    deg_q = n_q - 1

    # Check for zero divisor
    leading_q = q[..., -1]
    if torch.all(leading_q == 0):
        raise DegreeError("Cannot divide by zero polynomial")

    # If dividend degree < divisor degree, quotient is 0, remainder is dividend
    if deg_p < deg_q:
        zero_shape = list(p.shape)
        zero_shape[-1] = 1
        zero_coeffs = torch.zeros(zero_shape, dtype=p.dtype, device=p.device)
        from ._polynomial import polynomial

        return polynomial(zero_coeffs), p

    # Get batch shapes
    p_batch = p.shape[:-1]
    q_batch = q.shape[:-1]

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
    quot_flat, rem_flat = torch.ops.torchscience.polynomial_divmod(
        p_flat, q_flat
    )

    # Compute output sizes
    quot_len = n_p - n_q + 1
    rem_len = max(n_q - 1, 1)

    # Reshape output back to broadcast batch dimensions
    if len(broadcast_batch) == 0:
        quot_result = quot_flat.reshape(quot_len)
        rem_result = rem_flat.reshape(rem_len)
    else:
        quot_result = quot_flat.reshape(*broadcast_batch, quot_len)
        rem_result = rem_flat.reshape(*broadcast_batch, rem_len)

    from ._polynomial import polynomial

    return polynomial(quot_result), polynomial(rem_result)
