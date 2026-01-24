"""Hartley transform implementation."""

from typing import Literal

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


def hartley_transform(
    input: Tensor,
    *,
    dim: int | tuple[int, ...] = -1,
    n: int | tuple[int, ...] | None = None,
    norm: Literal["forward", "backward", "ortho"] = "backward",
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
    r"""Compute the discrete Hartley transform of a signal.

    The Hartley transform is defined as:

    .. math::
        H[k] = \text{Re}(F[k]) - \text{Im}(F[k])

    where :math:`F[k]` is the discrete Fourier transform. The Hartley transform
    is real-to-real and is self-inverse (applying it twice with ortho normalization
    recovers the original signal).

    This implementation wraps PyTorch's FFT with additional support for
    padding, windowing, and multi-dimensional transforms.

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Must be real-valued (complex input raises
        an error).
    dim : int or tuple of int, optional
        The dimension(s) along which to compute the transform.
        If a tuple, computes a multi-dimensional Hartley transform.
        Default: ``-1`` (last dimension).
    n : int or tuple of int, optional
        Signal length(s). If given, the input will either be padded or
        truncated to this length before computing the transform.
        For multi-dimensional transforms, must be a tuple with the same
        length as ``dim``.
        Default: ``None`` (use input size along ``dim``).
    norm : str, optional
        Normalization mode. One of:

        - ``'backward'``: No normalization on forward, divide by n on inverse.
        - ``'ortho'``: Normalize by 1/sqrt(n) on both forward and inverse.
          This makes the transform self-inverse.
        - ``'forward'``: Divide by n on forward, no normalization on inverse.

        Default: ``'backward'``.
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
        Output tensor. Must have the correct shape and dtype (real).
        Default: ``None`` (allocate new tensor).

    Returns
    -------
    Tensor
        The Hartley transform of the input. Always real-valued with the same
        dtype as the input.
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.

    Raises
    ------
    ValueError
        If the input tensor is complex-valued.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([1., 2., 3., 4.])
    >>> H = hartley_transform(x)
    >>> H.shape
    torch.Size([4])
    >>> H.dtype
    torch.float32

    Relationship to FFT (H = Re(F) - Im(F)):

    >>> x = torch.randn(32)
    >>> H = hartley_transform(x)
    >>> F = torch.fft.fft(x)
    >>> torch.allclose(H, F.real - F.imag)
    True

    Self-inverse property (with ortho normalization):

    >>> x = torch.randn(32)
    >>> H = hartley_transform(x, norm="ortho")
    >>> x_rec = hartley_transform(H, norm="ortho")
    >>> torch.allclose(x, x_rec, atol=1e-5)
    True

    With padding to get more frequency resolution:

    >>> x = torch.randn(64)
    >>> H = hartley_transform(x, n=128)
    >>> H.shape
    torch.Size([128])

    Multi-dimensional transform:

    >>> x = torch.randn(8, 16, 32)
    >>> H = hartley_transform(x, dim=(-2, -1))
    >>> H.shape
    torch.Size([8, 16, 32])

    Notes
    -----
    **Self-Inverse Property:**

    The Hartley transform is its own inverse when using ortho normalization:

    .. math::
        x = H(H(x))

    This is because the Hartley transform matrix is symmetric and orthogonal.

    **Relationship to Fourier Transform:**

    The Hartley transform is closely related to the Fourier transform:

    .. math::
        H[k] = \text{Re}(F[k]) - \text{Im}(F[k])

    And conversely:

    .. math::
        F[k] = \frac{H[k] + H[N-k]}{2} - i\frac{H[k] - H[N-k]}{2}

    **Advantages:**

    - Real input produces real output (no complex arithmetic needed)
    - Self-inverse (simpler to implement and understand)
    - Same computational complexity as FFT

    **Windowing:**

    Applying a window function before the transform reduces spectral leakage.
    Common windows include:

    - ``torch.hann_window``: Good general-purpose window
    - ``torch.hamming_window``: Similar to Hann
    - ``torch.blackman_window``: Better sidelobe suppression

    **Implementation:**

    Uses PyTorch's FFT backend (cuFFT on CUDA, MKL/FFTW on CPU) and extracts
    the Hartley transform as H = Re(F) - Im(F).

    **Gradient Computation:**

    Gradients are computed analytically via torch.fft's autograd support.
    Second-order gradients are also supported through torchscience.pad.

    See Also
    --------
    fourier_transform : The discrete Fourier transform.
    torch.fft.fft : PyTorch's 1D FFT implementation.
    """
    # Check that input is real
    if input.is_complex():
        raise ValueError(
            "hartley_transform requires real input. "
            f"Got complex dtype {input.dtype}. "
            "Use fourier_transform for complex inputs."
        )

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
        # Expand window to broadcast along the transform dimension
        # Create shape for broadcasting
        window_shape = [1] * x.ndim
        window_shape[d] = -1
        window_expanded = window.view(*window_shape)
        x = x * window_expanded

    # Perform the FFT and compute Hartley transform: H = Re(F) - Im(F)
    if len(dim_tuple) == 1:
        # 1D FFT
        fft_result = torch.fft.fft(x, dim=dim_tuple[0], norm=norm)
    else:
        # nD FFT
        fft_result = torch.fft.fftn(x, dim=dim_tuple, norm=norm)

    # Hartley transform: H = Re(F) - Im(F)
    result = fft_result.real - fft_result.imag

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result
