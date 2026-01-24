"""sRGB to YUV color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_yuv(input: Tensor) -> Tensor:
    r"""Convert sRGB to YUV color space (BT.601 analog).

    Converts input colors from sRGB to YUV using the BT.601 standard
    for analog video signals.

    Mathematical Definition
    -----------------------
    The conversion uses the BT.601 matrix:

    .. math::
        Y &= 0.299 R + 0.587 G + 0.114 B \\
        U &= -0.147 R - 0.289 G + 0.436 B \\
        V &= 0.615 R - 0.515 G - 0.100 B

    Note: Unlike YCbCr, U and V are centered around 0 (no +0.5 offset).

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values in [0, 1].

    Returns
    -------
    Tensor, shape (..., 3)
        YUV color values where Y is in [0, 1], U is approximately
        [-0.436, 0.436], and V is approximately [-0.615, 0.615].

    Examples
    --------
    Convert white to YUV:

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_yuv(rgb)
    tensor([[1.0000, 0.0000, 0.0000]])

    Convert red to YUV:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.srgb_to_yuv(rgb)
    tensor([[0.2990, -0.1471, 0.6150]])

    See Also
    --------
    yuv_to_srgb : Inverse conversion from YUV to sRGB.
    srgb_to_ycbcr : Similar conversion with offset U/V (digital video).

    References
    ----------
    .. [1] ITU-R BT.601-7, "Studio encoding parameters of digital television
           for standard 4:3 and wide screen 16:9 aspect ratios"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_yuv: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_yuv(input)
