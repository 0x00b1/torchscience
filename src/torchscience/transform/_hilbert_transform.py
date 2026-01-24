"""Hilbert transform implementation."""

import torch
from torch import Tensor

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


def hilbert_transform(
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
    r"""Compute the Hilbert transform of a signal along specified dimensions.

    The Hilbert transform is defined as:

    .. math::
        \mathcal{H}[f](x) = \frac{1}{\pi} \text{PV} \int_{-\infty}^{\infty}
        \frac{f(t)}{t - x} \, dt

    where PV denotes the Cauchy principal value.

    For discrete signals, this is computed efficiently using the FFT by
    multiplying the frequency spectrum by :math:`-i \cdot \text{sign}(\omega)`.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Can be real or complex.
    dim : int or tuple of int, optional
        The dimension(s) along which to compute the transform.
        If a tuple, applies the Hilbert transform sequentially along each
        dimension.
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
        Use window functions from ``torch`` (e.g., ``torch.hann_window``).
        Default: ``None`` (no windowing).
    out : Tensor, optional
        Output tensor. Must have the correct shape.
        Default: ``None`` (allocate new tensor).

    Returns
    -------
    Tensor
        The Hilbert transform of the input.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.
        If input is real, output is real (the imaginary part is discarded).

    Examples
    --------
    Basic usage with a sine wave:

    >>> t = torch.linspace(0, 2 * torch.pi, 100)
    >>> x = torch.sin(t)  # sin(t)
    >>> h = hilbert_transform(x)
    >>> # H[sin(t)] ~ -cos(t) for positive frequencies
    >>> torch.allclose(h, -torch.cos(t), atol=0.1)
    True

    Transform along a specific dimension:

    >>> x = torch.randn(3, 100)
    >>> h = hilbert_transform(x, dim=1)
    >>> h.shape
    torch.Size([3, 100])

    Multi-dimensional Hilbert transform:

    >>> x = torch.randn(8, 16, 32)
    >>> h = hilbert_transform(x, dim=(-2, -1))
    >>> h.shape
    torch.Size([8, 16, 32])

    With reflection padding to reduce edge effects:

    >>> x = torch.randn(64)
    >>> h = hilbert_transform(x, n=128, padding_mode='reflect')
    >>> h.shape
    torch.Size([128])

    With new padding modes:

    >>> x = torch.randn(32)
    >>> h = hilbert_transform(x, n=64, padding_mode="linear")
    >>> h.shape
    torch.Size([64])

    With a window function:

    >>> x = torch.randn(100)
    >>> window = torch.hann_window(100)
    >>> h = hilbert_transform(x, window=window)

    Notes
    -----
    **Mathematical Properties:**

    - :math:`\mathcal{H}[\sin(\omega t)] = -\cos(\omega t)` (for positive :math:`\omega`)
    - :math:`\mathcal{H}[\cos(\omega t)] = \sin(\omega t)` (for positive :math:`\omega`)
    - :math:`\mathcal{H}[\mathcal{H}[f]] = -f` (involutory up to sign)
    - Energy preservation: :math:`\int |H[f]|^2 = \int |f|^2`
    - Linearity: :math:`\mathcal{H}[\alpha f + \beta g] = \alpha\mathcal{H}[f] + \beta\mathcal{H}[g]`

    **Padding Modes:**

    - ``'constant'``: Zero-padding (default). Simple but can introduce
      discontinuities at boundaries.
    - ``'reflect'``: Reduces edge effects by reflecting the signal. Good for
      non-periodic signals.
    - ``'reflect_odd'``: Antisymmetric reflection for odd-symmetric signals.
    - ``'replicate'``: Extends the signal with edge values.
    - ``'circular'``: Wraps around, assuming periodicity. Best for truly
      periodic signals.
    - ``'linear'``: Linear extrapolation from edge values.
    - ``'polynomial'``: Polynomial extrapolation of degree ``padding_order``.
    - ``'spline'``: Cubic spline extrapolation.
    - ``'smooth'``: C1-continuous extension (matches value and derivative).

    **Multi-dimensional Transform:**

    For multi-dimensional transforms, the Hilbert transform is applied
    sequentially along each specified dimension. This is the standard
    separable approach.

    **Windowing:**

    Applying a window function before the transform can reduce spectral
    leakage and edge effects. Common windows include:

    - ``torch.hann_window``: Good general-purpose window
    - ``torch.hamming_window``: Similar to Hann, slightly different shape
    - ``torch.blackman_window``: Better sidelobe suppression

    **Complex Input Behavior:**

    For complex inputs, the transform is applied linearly to both components:
    :math:`\mathcal{H}[a + ib] = \mathcal{H}[a] + i\mathcal{H}[b]`. This
    preserves conjugate symmetry: :math:`\mathcal{H}[\overline{f}] = \overline{\mathcal{H}[f]}`.

    **Analytic Signal:**

    The analytic signal is defined as :math:`z(t) = f(t) + i\mathcal{H}[f](t)`.
    It can be computed as:

    >>> analytic = x + 1j * hilbert_transform(x)

    **Implementation:**

    Uses FFT-based computation:

    1. Apply padding if needed (using specified mode)
    2. Apply window function if provided
    3. Compute FFT of input
    4. Multiply by frequency response :math:`h[k] = -i \cdot \text{sign}(\text{freq}[k])`
    5. Compute inverse FFT

    **Gradient Computation:**

    Gradients are computed analytically using the property that the Hilbert
    transform is anti-self-adjoint: :math:`\mathcal{H}^T = -\mathcal{H}`.
    Therefore, for a loss :math:`L`:

    .. math::
        \frac{\partial L}{\partial x} = -\mathcal{H}\left[\frac{\partial L}{\partial y}\right]

    where :math:`y = \mathcal{H}[x]`. Second-order gradients are also supported
    through torchscience.pad.

    Warnings
    --------
    - Edge effects: The discrete Hilbert transform assumes periodic boundary
      conditions. For non-periodic signals, use ``padding_mode='reflect'``
      or ``padding_mode='replicate'`` with ``n > input.size(dim)``.

    - The transform is not well-defined for DC components (frequency = 0).

    - When using windowing, the window must match the padded signal length,
      not the original input length.

    References
    ----------
    .. [1] F.W. King, "Hilbert Transforms," Cambridge University Press, 2009.

    .. [2] S.L. Hahn, "Hilbert Transforms in Signal Processing,"
           Artech House, 1996.

    See Also
    --------
    inverse_hilbert_transform : The inverse Hilbert transform.
    scipy.signal.hilbert : SciPy's Hilbert transform (returns analytic signal).
    """
    # Validate padding_mode
    if padding_mode not in _VALID_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {sorted(_VALID_PADDING_MODES)}, "
            f"got '{padding_mode}'"
        )

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

    # Apply Hilbert transform along each dimension
    result = _hilbert_1d(x, normalized_dims[0])
    for d in normalized_dims[1:]:
        result = _hilbert_1d(result, d)

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result


def _hilbert_1d(x: Tensor, dim: int) -> Tensor:
    """Apply 1D Hilbert transform along specified dimension.

    The Hilbert transform in the frequency domain is multiplication by
    -i * sign(freq), which corresponds to:
    - Multiply positive frequencies by -i
    - Multiply negative frequencies by +i
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
        Hilbert transform of x along the specified dimension.
    """
    n = x.shape[dim]

    # Handle complex input by applying transform to real and imaginary parts
    is_complex = x.is_complex()

    # Compute FFT
    fft_x = torch.fft.fft(x, dim=dim)

    # Create the frequency response: -i * sign(freq)
    # For a signal of length n:
    # - DC (index 0): multiply by 0
    # - Positive frequencies (index 1 to n//2-1 for even n, 1 to (n-1)//2 for odd n): multiply by -i
    # - Nyquist (index n//2, only for even n): multiply by 0
    # - Negative frequencies (index n//2+1 to n-1 for even n): multiply by +i

    # Build multiplier tensor
    h = torch.zeros(n, dtype=fft_x.dtype, device=fft_x.device)

    if n > 1:
        if n % 2 == 0:
            # Even length
            # Positive frequencies: indices 1 to n//2 - 1
            h[1 : n // 2] = -1j
            # Negative frequencies: indices n//2 + 1 to n - 1
            h[n // 2 + 1 :] = 1j
            # DC (index 0) and Nyquist (index n//2) are 0
        else:
            # Odd length
            # Positive frequencies: indices 1 to (n-1)//2
            h[1 : (n + 1) // 2] = -1j
            # Negative frequencies: indices (n+1)//2 to n - 1
            h[(n + 1) // 2 :] = 1j
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
