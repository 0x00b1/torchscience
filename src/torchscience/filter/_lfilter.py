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
    >>> from torchscience.filter import lfilter
    >>> # Simple lowpass filter
    >>> b = torch.tensor([0.5, 0.5])
    >>> a = torch.tensor([1.0, -0.3])
    >>> x = torch.randn(100)
    >>> y = lfilter(b, a, x)

    >>> # With initial conditions (for smooth continuation)
    >>> from torchscience.filter import lfilter_zi
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


def lfiltic(
    b: Tensor,
    a: Tensor,
    y: Tensor,
    x: Optional[Tensor] = None,
) -> Tensor:
    """
    Construct initial conditions for lfilter given past outputs and inputs.

    Given a linear filter with coefficients b and a, and given past outputs y
    and optional past inputs x, compute the initial conditions zi that would
    produce those outputs.

    Parameters
    ----------
    b : Tensor
        Numerator coefficients, shape (M,).
    a : Tensor
        Denominator coefficients, shape (N,). a[0] must be non-zero.
    y : Tensor
        Past outputs, most recent first. Shape (K,) where K >= max(M, N) - 1.
        y[0] is the most recent past output, y[1] is the one before that, etc.
    x : Tensor, optional
        Past inputs, most recent first. Shape (K,) where K >= max(M, N) - 1.
        x[0] is the most recent past input. If None, zeros are assumed.

    Returns
    -------
    zi : Tensor
        Initial conditions suitable for lfilter, shape (max(M, N) - 1,).

    Notes
    -----
    This function is useful for initializing a filter to continue filtering
    a signal from where a previous filtering operation left off.

    The initial conditions are computed to match the Direct Form II Transposed
    structure used by lfilter.

    Examples
    --------
    >>> import torch
    >>> from torchscience.filter import lfilter, lfiltic
    >>> b = torch.tensor([0.1, 0.2, 0.1])
    >>> a = torch.tensor([1.0, -0.5, 0.1])
    >>> # Filter in two parts
    >>> x = torch.randn(100)
    >>> y1 = lfilter(b, a, x[:50])
    >>> zi = lfiltic(b, a, y1[-2:].flip(0), x[48:50].flip(0))
    >>> y2 = lfilter(b, a, x[50:], zi=zi)[0]
    """
    # Ensure 1-D coefficient tensors
    b = torch.atleast_1d(b)
    a = torch.atleast_1d(a)
    y = torch.atleast_1d(y)

    if b.ndim != 1:
        raise ValueError("b must be a 1-D tensor")
    if a.ndim != 1:
        raise ValueError("a must be a 1-D tensor")
    if y.ndim != 1:
        raise ValueError("y must be a 1-D tensor")

    # Determine output dtype
    dtype = torch.promote_types(b.dtype, torch.promote_types(a.dtype, y.dtype))
    if not dtype.is_floating_point:
        dtype = torch.float64

    device = y.device
    b = b.to(dtype=dtype, device=device)
    a = a.to(dtype=dtype, device=device)
    y = y.to(dtype=dtype)

    # Normalize by a[0]
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    n_b = b.shape[0]
    n_a = a.shape[0]
    n_coef = max(n_b, n_a)
    n_delays = n_coef - 1

    if n_delays == 0:
        return torch.tensor([], dtype=dtype, device=device)

    # Pad coefficients to same length
    if n_b < n_coef:
        b = F.pad(b, (0, n_coef - n_b))
    if n_a < n_coef:
        a = F.pad(a, (0, n_coef - n_a))

    # Handle x
    if x is None:
        x = torch.zeros(n_delays, dtype=dtype, device=device)
    else:
        x = torch.atleast_1d(x).to(dtype=dtype, device=device)

    # Pad y and x if necessary
    if len(y) < n_delays:
        y = torch.cat(
            [y, torch.zeros(n_delays - len(y), dtype=dtype, device=device)]
        )
    if len(x) < n_delays:
        x = torch.cat(
            [x, torch.zeros(n_delays - len(x), dtype=dtype, device=device)]
        )

    # Compute initial conditions
    # For Direct Form II Transposed, the state equations are:
    # z[k] = b[k+1]*x - a[k+1]*y + z[k+1]  for k = 0, ..., n_delays-2
    # z[n_delays-1] = b[n_coef-1]*x - a[n_coef-1]*y
    zi = torch.zeros(n_delays, dtype=dtype, device=device)

    # Work backwards from the last state
    for k in range(n_delays - 1, -1, -1):
        # Accumulate contributions from past samples
        zi_k = torch.tensor(0.0, dtype=dtype, device=device)
        for j in range(k, n_delays):
            idx = j - k
            if idx < len(y):
                zi_k = zi_k + b[j + 1] * x[idx] - a[j + 1] * y[idx]
        zi[k] = zi_k

    return zi
