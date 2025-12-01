"""Stencil application to fields."""

from __future__ import annotations

from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torchscience.differentiation._stencil import FiniteDifferenceStencil


def apply_stencil(
    stencil: FiniteDifferenceStencil,
    field: Tensor,
    dx: Union[float, Tuple[float, ...]] = 1.0,
    dim: Tuple[int, ...] | None = None,
    boundary: str = "replicate",
) -> Tensor:
    """Apply finite difference stencil to a field.

    Parameters
    ----------
    stencil : FiniteDifferenceStencil
        Stencil to apply.
    field : Tensor
        Input field with shape (..., *spatial_dims).
    dx : float or tuple of float
        Grid spacing. Scalar applies to all dimensions.
    dim : tuple of int, optional
        Spatial dimensions of the field. Default uses trailing stencil.ndim dimensions.
    boundary : str
        Boundary handling: "replicate", "zeros", "reflect", "circular", "valid"

    Returns
    -------
    Tensor
        Derivative field. Same shape as input unless boundary="valid".

    Examples
    --------
    >>> x = torch.linspace(0, 1, 11)
    >>> f = x**2
    >>> stencil = finite_difference_stencil(derivative=2, accuracy=2)
    >>> result = apply_stencil(stencil, f, dx=0.1)  # Second derivative
    """
    ndim = stencil.ndim

    # Determine spatial dimensions
    if dim is None:
        # Use trailing dimensions
        spatial_dims = tuple(range(-ndim, 0))
    else:
        spatial_dims = tuple(dim)

    # Handle dx normalization
    if isinstance(dx, (int, float)):
        dx_tuple = (float(dx),) * ndim
    else:
        dx_tuple = tuple(float(d) for d in dx)
        if len(dx_tuple) != ndim:
            raise ValueError(
                f"dx has {len(dx_tuple)} elements but stencil has {ndim} dimensions"
            )

    # Compute scale factor: product of 1/dx_i^order_i for each dimension
    scale = 1.0
    for i, deriv_order in enumerate(stencil.derivative):
        if deriv_order > 0:
            scale /= dx_tuple[i] ** deriv_order

    # Get dense kernel from stencil
    kernel = stencil.to_dense()  # Shape: (1, 1, *kernel_spatial)
    kernel = kernel * scale

    # Convert kernel to field's dtype
    kernel = kernel.to(dtype=field.dtype, device=field.device)

    # Reshape field for convolution
    # Conv expects (batch, channels, *spatial) format
    # We treat all non-spatial dimensions as batch
    original_shape = field.shape
    spatial_shape = original_shape[-ndim:] if ndim > 0 else ()
    batch_shape = original_shape[:-ndim] if ndim > 0 else original_shape

    # Flatten batch dimensions
    if len(batch_shape) == 0:
        # Pure spatial tensor - add batch and channel dims
        field_conv = field.unsqueeze(0).unsqueeze(0)
    elif len(batch_shape) == 1:
        # Already has batch dim - add channel dim
        field_conv = field.unsqueeze(1)
    else:
        # Multiple batch dims - flatten them
        batch_size = 1
        for s in batch_shape:
            batch_size *= s
        field_conv = field.reshape(batch_size, *spatial_shape).unsqueeze(1)

    # Determine padding
    kernel_shape = kernel.shape[2:]  # Remove batch and channel dims

    if boundary == "valid":
        padding = 0
    else:
        # Compute padding to maintain same size
        # For a kernel of size k, we need (k-1)/2 padding on each side
        # But for asymmetric kernels, we need different left/right padding
        padding_list = []
        for k in kernel_shape:
            padding_list.append((k - 1) // 2)
        padding = tuple(padding_list)

    # Apply boundary condition via padding
    if boundary != "valid":
        # Compute explicit padding for each dimension
        pad_amounts = []
        for i, k in enumerate(reversed(kernel_shape)):
            left_pad = (k - 1) // 2
            right_pad = k - 1 - left_pad
            pad_amounts.extend([left_pad, right_pad])

        # Map boundary mode to F.pad mode
        pad_mode_map = {
            "replicate": "replicate",
            "zeros": "constant",
            "reflect": "reflect",
            "circular": "circular",
        }

        if boundary not in pad_mode_map:
            raise ValueError(
                f"Unknown boundary mode '{boundary}'. "
                f"Supported modes: replicate, zeros, reflect, circular, valid"
            )

        pad_mode = pad_mode_map[boundary]
        field_conv = F.pad(field_conv, pad_amounts, mode=pad_mode)
        conv_padding = 0
    else:
        conv_padding = 0

    # Apply convolution based on dimensionality
    if ndim == 1:
        result_conv = F.conv1d(field_conv, kernel, padding=conv_padding)
    elif ndim == 2:
        result_conv = F.conv2d(field_conv, kernel, padding=conv_padding)
    elif ndim == 3:
        result_conv = F.conv3d(field_conv, kernel, padding=conv_padding)
    else:
        raise ValueError(
            f"Stencil has {ndim} dimensions, but only 1D, 2D, 3D are supported"
        )

    # Reshape result back to original format
    result_spatial_shape = result_conv.shape[2:]

    if len(batch_shape) == 0:
        # Remove batch and channel dims
        result = result_conv.squeeze(1).squeeze(0)
    elif len(batch_shape) == 1:
        # Remove channel dim
        result = result_conv.squeeze(1)
    else:
        # Unflatten batch dims
        result = result_conv.squeeze(1).reshape(
            *batch_shape, *result_spatial_shape
        )

    return result
