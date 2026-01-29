"""Inverse Hilbert transform implementation."""

from __future__ import annotations

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401
from torchscience.pad import PaddingMode, pad

# Valid padding modes for validation
_VALID_PADDING_MODES = {
    "constant",
    "replicate",
    "reflect",
    "reflect_odd",
    "circular",
    "linear",
    "polynomial",
    "spline",
    "smooth",
}

# Padding modes supported by C++ backend
_CPP_PADDING_MODES = {
    "constant": 0,
    "reflect": 1,
    "replicate": 2,
    "circular": 3,
}


def _can_use_cpp_backend(
    dim: int | tuple[int, ...],
    padding: int
    | tuple[int, int]
    | tuple[int, ...]
    | tuple[tuple[int, int], ...]
    | None,
    padding_mode: str,
    padding_order: int,
    out: Tensor | None,
) -> bool:
    """Check if we can use the C++ backend for this call."""
    # C++ backend only supports single dimension
    if not isinstance(dim, int):
        return False

    # C++ backend doesn't support explicit padding parameter
    if padding is not None:
        return False

    # C++ backend only supports basic padding modes
    if padding_mode not in _CPP_PADDING_MODES:
        return False

    # C++ backend doesn't support out parameter
    if out is not None:
        return False

    return True


def inverse_hilbert_transform(
    input: Tensor,
    *,
    dim: int | tuple[int, ...] = -1,
    n: int | tuple[int, ...] | None = None,
    padding: int
    | tuple[int, int]
    | tuple[int, ...]
    | tuple[tuple[int, int], ...]
    | None = None,
    padding_mode: PaddingMode = "constant",
    padding_value: float = 0.0,
    padding_order: int = 1,
    window: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    r"""Compute the inverse Hilbert transform of a signal.

    The inverse Hilbert transform recovers the original signal from its
    Hilbert transform. It is defined as:

    .. math::
        \mathcal{H}^{-1}[f] = -\mathcal{H}[f]

    This follows from the property that :math:`\mathcal{H}[\mathcal{H}[f]] = -f`.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Can be real or complex.
    dim : int or tuple of int, optional
        The dimension(s) along which to compute the transform.
        If a tuple, applies the inverse Hilbert transform sequentially along
        each dimension.
        Default: ``-1`` (last dimension).
    n : int or tuple of int, optional
        Signal length(s). If given, the input will either be padded or
        truncated to this length before computing the transform.
        For multi-dimensional transforms, must be a tuple with the same
        length as ``dim``.
        Default: ``None`` (use input size along ``dim``).
    padding : int, tuple of int, or tuple of tuple of int, optional
        Explicit padding to apply before the transform. Accepts multiple formats:

        - ``int``: Same padding on both sides
        - ``(before, after)``: Asymmetric padding for single dimension
        - ``((before_0, after_0), (before_1, after_1), ...)``: Per-dimension pairs

        If ``n`` is also specified and larger than the padded size, additional
        padding will be applied to reach the target size.
        Default: ``None`` (no explicit padding).
    padding_mode : str, optional
        Padding mode when padding is needed. One of:

        - ``'constant'``: Pad with ``padding_value`` (default 0).
        - ``'reflect'``: Reflect the signal at boundaries.
        - ``'reflect_odd'``: Antisymmetric reflection.
        - ``'replicate'``: Replicate edge values.
        - ``'circular'``: Wrap around (periodic extension).
        - ``'linear'``: Linear extrapolation from edge.
        - ``'polynomial'``: Polynomial extrapolation of degree ``padding_order``.
        - ``'spline'``: Cubic spline extrapolation.
        - ``'smooth'``: C1-continuous extension (matches value and derivative).

        Default: ``'constant'``.
    padding_value : float, optional
        Fill value for ``'constant'`` padding mode. Ignored for other modes.
        Default: ``0.0``.
    padding_order : int, optional
        Polynomial order for ``'polynomial'`` padding mode.
        Default: ``1`` (linear).
    window : Tensor, optional
        Window function to apply before the transform. Must be 1-D with size
        matching the (possibly padded) signal length along ``dim``.
        For multi-dimensional transforms, windowing is only supported for
        single-dimension transforms.
        Default: ``None`` (no windowing).
    out : Tensor, optional
        Output tensor. Must have the correct shape.
        Default: ``None`` (allocate new tensor).

    Returns
    -------
    Tensor
        The inverse Hilbert transform of the input.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.

    Examples
    --------
    Verify that inverse undoes forward transform:

    >>> import torchscience.transform
    >>> x = torch.randn(100)
    >>> h = torchscience.transform.hilbert_transform(x)
    >>> x_recovered = torchscience.transform.inverse_hilbert_transform(h)
    >>> torch.allclose(x, x_recovered, atol=1e-5)
    True

    Transform along a specific dimension:

    >>> x = torch.randn(3, 100)
    >>> h = inverse_hilbert_transform(x, dim=1)
    >>> h.shape
    torch.Size([3, 100])

    Multi-dimensional inverse Hilbert transform:

    >>> x = torch.randn(8, 16, 32)
    >>> h = inverse_hilbert_transform(x, dim=(-2, -1))
    >>> h.shape
    torch.Size([8, 16, 32])

    With new padding modes:

    >>> x = torch.randn(32)
    >>> h = inverse_hilbert_transform(x, n=64, padding_mode="linear")
    >>> h.shape
    torch.Size([64])

    Notes
    -----
    **Mathematical Properties:**

    - :math:`\mathcal{H}^{-1}[\mathcal{H}[f]] = f`
    - :math:`\mathcal{H}[\mathcal{H}^{-1}[f]] = f`
    - :math:`\mathcal{H}^{-1}[f] = -\mathcal{H}[f]`

    **Complex Input Behavior:**

    For complex inputs, the transform is applied linearly to both components:
    :math:`\mathcal{H}^{-1}[a + ib] = \mathcal{H}^{-1}[a] + i\mathcal{H}^{-1}[b]`.

    **Multi-dimensional Transform:**

    For multi-dimensional transforms, the inverse Hilbert transform is applied
    sequentially along each specified dimension. This is the standard
    separable approach.

    **Implementation:**

    Uses the same FFT-based approach as the forward transform, but with
    the negated frequency response :math:`h^{-1}[k] = i \cdot \text{sign}(\text{freq}[k])`.

    **Gradient Computation:**

    Gradients are computed analytically. The adjoint of the inverse Hilbert
    transform is the forward Hilbert transform: :math:`(\mathcal{H}^{-1})^T = \mathcal{H}`.
    Therefore:

    .. math::
        \frac{\partial L}{\partial x} = \mathcal{H}\left[\frac{\partial L}{\partial y}\right]

    where :math:`y = \mathcal{H}^{-1}[x]`. Second-order gradients are also supported
    through torchscience.pad.

    References
    ----------
    .. [1] F.W. King, "Hilbert Transforms," Cambridge University Press, 2009.

    See Also
    --------
    hilbert_transform : The forward Hilbert transform.
    """
    # Validate padding_mode
    if padding_mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {sorted(_VALID_PADDING_MODES)}, "
            f"got '{padding_mode}'"
        )

    # Try to use C++ backend for simple cases
    if _can_use_cpp_backend(dim, padding, padding_mode, padding_order, out):
        # Single dimension, basic padding mode, no explicit padding
        assert isinstance(dim, int)

        # Normalize n for C++ backend
        n_param = n if isinstance(n, int) else -1 if n is None else n[0]

        # Normalize dimension for validation
        ndim = input.ndim
        norm_dim = dim if dim >= 0 else dim + ndim

        # Validate window before calling C++ to get proper Python errors
        if window is not None:
            if window.ndim != 1:
                raise ValueError(
                    f"window must be 1-D, got {window.ndim}-D tensor with shape {window.shape}"
                )
            # Compute expected size after padding
            input_size = input.size(norm_dim)
            expected_size = n_param if n_param > 0 else input_size
            if window.size(0) != expected_size:
                raise ValueError(
                    f"window size ({window.size(0)}) must match signal length along "
                    f"dimension {dim} ({expected_size})"
                )
            if window.device != input.device:
                raise RuntimeError(
                    f"window tensor must be on the same device as input tensor. "
                    f"Input is on {input.device}, window is on {window.device}."
                )

        return torch.ops.torchscience.inverse_hilbert_transform(
            input,
            n_param,
            dim,
            _CPP_PADDING_MODES[padding_mode],
            padding_value,
            window,
        )

    # Fall back to Python implementation for advanced features
    return _inverse_hilbert_transform_python(
        input,
        dim=dim,
        n=n,
        padding=padding,
        padding_mode=padding_mode,
        padding_value=padding_value,
        padding_order=padding_order,
        window=window,
        out=out,
    )


def _inverse_hilbert_transform_python(
    input: Tensor,
    *,
    dim: int | tuple[int, ...] = -1,
    n: int | tuple[int, ...] | None = None,
    padding: int
    | tuple[int, int]
    | tuple[int, ...]
    | tuple[tuple[int, int], ...]
    | None = None,
    padding_mode: PaddingMode = "constant",
    padding_value: float = 0.0,
    padding_order: int = 1,
    window: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Pure Python implementation for advanced features."""
    # Normalize dim to tuple
    if isinstance(dim, int):
        dim_tuple = (dim,)
    else:
        dim_tuple = tuple(dim)

    # Normalize n to tuple if provided
    if n is not None:
        if isinstance(n, int):
            n_tuple: tuple[int, ...] | None = (n,)
        else:
            n_tuple = tuple(n)
        # Validate n and dim have same length
        if len(n_tuple) != len(dim_tuple):
            raise ValueError(
                f"n tuple length ({len(n_tuple)}) must match dim tuple length "
                f"({len(dim_tuple)})"
            )
    else:
        n_tuple = None

    # Normalize dimensions (handle negative indices)
    ndim = input.ndim
    normalized_dims = tuple(d if d >= 0 else d + ndim for d in dim_tuple)

    # Get current sizes along transform dimensions
    current_sizes = tuple(input.shape[d] for d in normalized_dims)

    # Determine target sizes
    if n_tuple is not None:
        target_sizes = n_tuple
    else:
        target_sizes = current_sizes

    # Apply explicit padding first if specified
    x = input
    if padding is not None:
        # Normalize padding format for the specified dimensions
        if isinstance(padding, int):
            # Same padding on both sides for all dims
            pad_spec: tuple[tuple[int, int], ...] = tuple(
                (padding, padding) for _ in dim_tuple
            )
        elif isinstance(padding, tuple):
            if padding and isinstance(padding[0], tuple):
                # NumPy-style: ((before_0, after_0), ...)
                pad_spec = tuple(padding)  # type: ignore
            elif len(padding) == 2 and len(dim_tuple) == 1:
                # (before, after) for single dim
                pad_spec = (tuple(padding),)  # type: ignore
            else:
                # PyTorch-style flat: (left_n, right_n, ..., left_0, right_0)
                # Convert to per-dim pairs
                pad_pairs = []
                for i in range(0, len(padding), 2):
                    if i + 1 < len(padding):
                        pad_pairs.append((padding[i], padding[i + 1]))
                pad_spec = tuple(reversed(pad_pairs))
        else:
            pad_spec = tuple((padding, padding) for _ in dim_tuple)

        # Apply padding using torchscience.pad
        # Build padding for torch format (reversed, flattened)
        torch_pad: list[int] = []
        for before, after in reversed(pad_spec):
            torch_pad.extend([before, after])

        x = pad(
            x,
            tuple(torch_pad),
            mode=padding_mode,
            value=padding_value,
            order=padding_order,
        )

        # Update current sizes after explicit padding
        current_sizes = tuple(x.shape[d] for d in normalized_dims)

    # Handle size changes (via n parameter)
    if n_tuple is not None:
        needs_processing = False
        for curr, target in zip(current_sizes, target_sizes):
            if curr != target:
                needs_processing = True
                break

        if needs_processing:
            # Process each dimension
            for i, (d, curr, target) in enumerate(
                zip(normalized_dims, current_sizes, target_sizes)
            ):
                if curr < target:
                    # Need to pad - pad at the end (right side)
                    pad_amount = target - curr
                    x = pad(
                        x,
                        (0, pad_amount),
                        mode=padding_mode,
                        value=padding_value,
                        dim=d,
                        order=padding_order,
                    )
                elif curr > target:
                    # Need to truncate
                    x = x.narrow(d, 0, target)

    # Apply window if provided
    if window is not None:
        if len(dim_tuple) != 1:
            raise ValueError(
                "Windowing is only supported for single-dimension transforms. "
                f"Got dim={dim} which has {len(dim_tuple)} dimensions."
            )
        # Validate window is 1-D
        if window.ndim != 1:
            raise ValueError(
                f"window must be 1-D, got {window.ndim}-D tensor with shape {window.shape}"
            )
        # Validate window size matches signal length
        d = normalized_dims[0]
        expected_size = x.shape[d]
        if window.size(0) != expected_size:
            raise ValueError(
                f"window size ({window.size(0)}) must match signal length along "
                f"dimension {dim_tuple[0]} ({expected_size})"
            )
        # Check device match
        if window.device != x.device:
            raise RuntimeError(
                f"window tensor must be on the same device as input tensor. "
                f"Input is on {x.device}, window is on {window.device}."
            )
        # Expand window to broadcast along the transform dimension
        # Create shape for broadcasting
        window_shape = [1] * x.ndim
        window_shape[d] = -1
        window_expanded = window.view(*window_shape)
        x = x * window_expanded

    # Apply inverse Hilbert transform along each dimension
    result = _inverse_hilbert_1d(x, normalized_dims[0])
    for d in normalized_dims[1:]:
        result = _inverse_hilbert_1d(result, d)

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result


def _inverse_hilbert_1d(x: Tensor, dim: int) -> Tensor:
    """Apply 1D inverse Hilbert transform along specified dimension.

    The inverse Hilbert transform is the negative of the Hilbert transform.
    In the frequency domain, it's multiplication by +i * sign(freq), which
    corresponds to:
    - Multiply positive frequencies by +i
    - Multiply negative frequencies by -i
    - Leave DC and Nyquist unchanged (multiply by 0)

    Parameters
    ----------
    x : Tensor
        Input tensor.
    dim : int
        Dimension along which to apply the transform (must be normalized).

    Returns
    -------
    Tensor
        Inverse Hilbert transform of x along the specified dimension.
    """
    n = x.shape[dim]

    # Handle complex input by applying transform to real and imaginary parts
    is_complex = x.is_complex()

    # Compute FFT
    fft_x = torch.fft.fft(x, dim=dim)

    # Create the frequency response: +i * sign(freq) (negated from forward transform)
    # For a signal of length n:
    # - DC (index 0): multiply by 0
    # - Positive frequencies (index 1 to n//2-1 for even n): multiply by +i
    # - Nyquist (index n//2, only for even n): multiply by 0
    # - Negative frequencies (index n//2+1 to n-1): multiply by -i

    # Build multiplier tensor
    h = torch.zeros(n, dtype=fft_x.dtype, device=fft_x.device)

    if n > 1:
        if n % 2 == 0:
            # Even length
            # Positive frequencies: indices 1 to n//2 - 1
            h[1 : n // 2] = 1j
            # Negative frequencies: indices n//2 + 1 to n - 1
            h[n // 2 + 1 :] = -1j
            # DC (index 0) and Nyquist (index n//2) are 0
        else:
            # Odd length
            # Positive frequencies: indices 1 to (n-1)//2
            h[1 : (n + 1) // 2] = 1j
            # Negative frequencies: indices (n+1)//2 to n - 1
            h[(n + 1) // 2 :] = -1j
            # DC (index 0) is 0

    # Expand h to broadcast along the transform dimension
    shape = [1] * fft_x.ndim
    shape[dim] = n
    h = h.view(*shape)

    # Apply frequency response
    fft_result = fft_x * h

    # Inverse FFT
    result = torch.fft.ifft(fft_result, dim=dim)

    # For real input, return real output; for complex input, return complex output
    if not is_complex:
        result = result.real

    return result
