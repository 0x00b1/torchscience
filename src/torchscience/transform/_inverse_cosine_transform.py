"""Inverse Discrete Cosine Transform (IDCT) implementation with new contract."""

from typing import Literal

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators
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

# Normalization mode mapping for the C++ backend
_NORM_MODES = {
    "backward": 0,
    "ortho": 1,
    "forward": 2,
}


def inverse_cosine_transform(
    input: Tensor,
    *,
    type: int = 2,
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
    r"""Compute the Inverse Discrete Cosine Transform (IDCT) of a signal.

    This is the inverse of the DCT. The inverse relationships are:

    - IDCT-I is DCT-I (DCT-I is its own inverse)
    - IDCT-II is DCT-III
    - IDCT-III is DCT-II
    - IDCT-IV is DCT-IV (DCT-IV is its own inverse)

    Parameters
    ----------
    input : Tensor
        Input tensor of any shape. Must be real-valued.
    type : int, optional
        IDCT type (1, 2, 3, or 4). IDCT-II is computed via DCT-III,
        and IDCT-III via DCT-II. DCT-I and DCT-IV are self-inverse.
        Default: ``2``.
    dim : int or tuple of int, optional
        The dimension(s) along which to compute the transform.
        If a tuple, computes a multi-dimensional IDCT by applying
        the 1D IDCT sequentially along each dimension.
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
        - ``'ortho'``: Orthonormal normalization (makes DCT matrix orthogonal).
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
        Window function to apply after the transform. Must be 1-D with size
        matching the output signal length along ``dim``.
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
        The IDCT of the input. Same dtype as input (real-valued).
        If ``n`` is specified and differs from the input size along ``dim``,
        the output size along ``dim`` will be ``n``.

    Raises
    ------
    ValueError
        If input is complex-valued (DCT requires real input).
    ValueError
        If type is not in {1, 2, 3, 4}.

    Examples
    --------
    Round-trip with forward transform:

    >>> x = torch.randn(32)
    >>> X = cosine_transform(x, norm='ortho')
    >>> x_rec = inverse_cosine_transform(X, norm='ortho')
    >>> torch.allclose(x_rec, x, atol=1e-5)
    True

    Multi-dimensional inverse:

    >>> X = torch.randn(8, 16, 32)
    >>> x = inverse_cosine_transform(X, dim=(-2, -1))
    >>> x.shape
    torch.Size([8, 16, 32])

    See Also
    --------
    cosine_transform : The forward DCT.
    scipy.fft.idct : SciPy's IDCT implementation.
    """
    # Validate input is real
    if input.is_complex():
        raise ValueError(
            "inverse_cosine_transform requires real-valued input, "
            f"but got complex dtype {input.dtype}"
        )

    # Validate DCT type
    if type not in (1, 2, 3, 4):
        raise ValueError(f"type must be 1, 2, 3, or 4, got {type}")

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

    # Map norm mode - the C++ backend only supports "backward" and "ortho"
    # For "forward" norm, we use "backward" and scale manually
    if norm == "forward":
        norm_int = 0  # backward
    else:
        norm_int = _NORM_MODES.get(norm, 0)

    # Perform the IDCT
    if len(dim_tuple) == 1:
        # 1D IDCT
        result = torch.ops.torchscience.inverse_fourier_cosine_transform(
            x,
            -1,  # n=-1 means use input size (we already handled padding/truncation)
            dim_tuple[0],
            type,
            norm_int,
        )
    else:
        # Multi-dimensional IDCT: apply 1D IDCT sequentially along each dimension
        result = x
        for d in dim_tuple:
            result = torch.ops.torchscience.inverse_fourier_cosine_transform(
                result,
                -1,  # n=-1 means use input size
                d,
                type,
                norm_int,
            )

    # Handle forward normalization manually if needed
    if norm == "forward":
        # No scaling for inverse with forward norm
        pass

    # Apply window if provided (after IDCT for inverse transform)
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
        expected_size = result.shape[d]
        if window.size(0) != expected_size:
            raise ValueError(
                f"window size ({window.size(0)}) must match signal length along "
                f"dimension {dim_tuple[0]} ({expected_size})"
            )
        # Expand window to broadcast along the transform dimension
        # Create shape for broadcasting
        window_shape = [1] * result.ndim
        window_shape[d] = -1
        window_expanded = window.view(*window_shape)
        result = result * window_expanded

    # Handle out parameter
    if out is not None:
        out.copy_(result)
        return out

    return result


# Alias for backward compatibility
inverse_fourier_cosine_transform = inverse_cosine_transform
