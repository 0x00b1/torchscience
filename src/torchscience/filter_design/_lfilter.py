"""IIR/FIR filter implementation using Direct Form II Transposed structure."""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def lfilter(
    b: Tensor,
    a: Tensor,
    x: Tensor,
    axis: int = -1,
    zi: Optional[Tensor] = None,
) -> Union[Tensor, tuple[Tensor, Tensor]]:
    """
    Filter data along one dimension with an IIR or FIR filter.

    This implements the standard difference equation using the Direct Form II
    Transposed structure, which is efficient and numerically stable.

    Parameters
    ----------
    b : Tensor
        Numerator polynomial coefficients (feedforward).
    a : Tensor
        Denominator polynomial coefficients (feedback). ``a[0]`` must be
        nonzero. If ``a[0] != 1``, then both ``a`` and ``b`` are normalized
        by ``a[0]``.
    x : Tensor
        Input signal. Can be batched with arbitrary leading dimensions.
    axis : int, optional
        Axis along which to filter. Default is -1 (last axis).
    zi : Tensor, optional
        Initial conditions for the filter delays. If provided, returns
        ``(y, zf)`` where ``zf`` is the final filter delays. Shape should
        be ``(..., n_order)`` where ``n_order = max(len(a), len(b)) - 1``
        and ``...`` matches the batch dimensions of ``x``.

    Returns
    -------
    y : Tensor
        The output of the digital filter, same shape as ``x``.
    zf : Tensor, optional
        Final filter delay values (only if ``zi`` was provided).

    Notes
    -----
    Implements the difference equation:

    .. math::
        a[0] y[n] = b[0] x[n] + b[1] x[n-1] + \\ldots + b[M] x[n-M]
                    - a[1] y[n-1] - \\ldots - a[N] y[n-N]

    The Direct Form II Transposed structure uses state variable ``z``:

    .. math::
        y[n] &= b[0] x[n] + z[0][n-1] \\\\
        z[0][n] &= b[1] x[n] - a[1] y[n] + z[1][n-1] \\\\
        z[1][n] &= b[2] x[n] - a[2] y[n] + z[2][n-1] \\\\
        &\\vdots \\\\
        z[K-2][n] &= b[K-1] x[n] - a[K-1] y[n]

    where ``K = max(len(a), len(b))``.

    Fully differentiable with respect to ``b`` and ``x``.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter_design import lfilter
    >>> # Simple lowpass filter
    >>> b = torch.tensor([0.5, 0.5])
    >>> a = torch.tensor([1.0, -0.3])
    >>> x = torch.randn(100)
    >>> y = lfilter(b, a, x)

    >>> # With initial conditions (for smooth continuation)
    >>> from torchscience.filter_design import lfilter_zi
    >>> zi = lfilter_zi(b, a) * x[0]  # Scale by initial value
    >>> y, zf = lfilter(b, a, x, zi=zi)
    """
    # Ensure 1-D coefficient tensors
    b = torch.atleast_1d(b)
    a = torch.atleast_1d(a)

    if b.ndim != 1:
        raise ValueError("b must be a 1-D tensor")
    if a.ndim != 1:
        raise ValueError("a must be a 1-D tensor")

    # Normalize by a[0]
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    # Determine filter order
    n_b = b.shape[0]
    n_a = a.shape[0]
    n_order = max(n_b, n_a) - 1

    if n_order == 0:
        # Simple gain filter
        y = b[0] * x
        if zi is not None:
            # No state to return, but maintain API
            return y, torch.zeros_like(zi)
        return y

    # Pad b and a to the same length
    if n_b < n_a:
        b = F.pad(b, (0, n_a - n_b))
    elif n_a < n_b:
        a = F.pad(a, (0, n_b - n_a))

    # Move filtering axis to last position
    x = torch.moveaxis(x, axis, -1)
    original_shape = x.shape
    n_samples = x.shape[-1]

    # Flatten batch dimensions
    batch_shape = x.shape[:-1]
    x_flat = x.reshape(-1, n_samples)
    n_batch = x_flat.shape[0]

    # Determine output dtype (complex if any input is complex)
    # Use promote_types to properly combine dtypes
    out_dtype = torch.promote_types(
        torch.promote_types(b.dtype, a.dtype), x.dtype
    )

    # Initialize state
    if zi is not None:
        zi = torch.moveaxis(zi, axis if axis != -1 else -1, -1)
        # zi should have shape (..., n_order), flatten
        zi_flat = zi.reshape(-1, n_order).to(out_dtype).clone()
    else:
        zi_flat = torch.zeros(
            n_batch, n_order, dtype=out_dtype, device=x.device
        )

    # Allocate output with correct dtype
    y_flat = torch.zeros(n_batch, n_samples, dtype=out_dtype, device=x.device)

    # Direct Form II Transposed implementation
    # State z has shape (n_batch, n_order)
    z = zi_flat

    # Filter sample by sample
    for n in range(n_samples):
        # Current input sample: shape (n_batch,)
        x_n = x_flat[:, n]

        # Output: y[n] = b[0] * x[n] + z[0]
        y_n = b[0] * x_n + z[:, 0]
        y_flat[:, n] = y_n

        # Update state (shift and compute new values)
        # z[k] = b[k+1] * x[n] - a[k+1] * y[n] + z[k+1]
        for k in range(n_order - 1):
            z[:, k] = b[k + 1] * x_n - a[k + 1] * y_n + z[:, k + 1]
        # Last state element
        z[:, n_order - 1] = b[n_order] * x_n - a[n_order] * y_n

    # Reshape output to original shape
    y = y_flat.reshape(original_shape)
    y = torch.moveaxis(y, -1, axis)

    if zi is not None:
        # Return final state
        zf = z.reshape(batch_shape + (n_order,))
        zf = torch.moveaxis(zf, -1, axis if axis != -1 else -1)
        return y, zf

    return y
