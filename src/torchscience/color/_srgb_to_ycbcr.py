"""sRGB to YCbCr color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_ycbcr(input: Tensor) -> Tensor:
    r"""Convert sRGB to YCbCr color space (ITU-R BT.601).

    Converts input colors from sRGB to YCbCr using the BT.601 standard
    commonly used in JPEG and video compression.

    Mathematical Definition
    -----------------------
    The conversion uses the BT.601 matrix:

    .. math::
        Y  &= 0.299 R + 0.587 G + 0.114 B \\
        Cb &= -0.168736 R - 0.331264 G + 0.5 B + 0.5 \\
        Cr &= 0.5 R - 0.418688 G - 0.081312 B + 0.5

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values in [0, 1].

    Returns
    -------
    Tensor, shape (..., 3)
        YCbCr color values where Y is in [0, 1] and Cb, Cr are in [0, 1]
        (centered at 0.5).

    Examples
    --------
    Convert white to YCbCr:

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_ycbcr(rgb)
    tensor([[1.0000, 0.5000, 0.5000]])

    Convert red to YCbCr:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.srgb_to_ycbcr(rgb)
    tensor([[0.2990, 0.3313, 1.0000]])

    See Also
    --------
    ycbcr_to_srgb : Inverse conversion from YCbCr to sRGB.

    References
    ----------
    .. [1] ITU-R BT.601-7, "Studio encoding parameters of digital television
           for standard 4:3 and wide screen 16:9 aspect ratios"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_ycbcr: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_ycbcr(input)
