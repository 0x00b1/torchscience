"""YUV to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def yuv_to_srgb(input: Tensor) -> Tensor:
    r"""Convert YUV to sRGB color space (BT.601 analog).

    Converts input colors from YUV to sRGB using the inverse BT.601
    transformation for analog video signals.

    Mathematical Definition
    -----------------------
    The inverse conversion is:

    .. math::
        R &= Y + 1.140 V \\
        G &= Y - 0.395 U - 0.581 V \\
        B &= Y + 2.032 U

    Note: Unlike YCbCr, U and V are already centered around 0.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        YUV color values where Y is in [0, 1], U is approximately
        [-0.436, 0.436], and V is approximately [-0.615, 0.615].

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values in [0, 1].

    Examples
    --------
    Convert white from YUV to sRGB:

    >>> yuv = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.yuv_to_srgb(yuv)
    tensor([[1.0000, 1.0000, 1.0000]])

    See Also
    --------
    srgb_to_yuv : Forward conversion from sRGB to YUV.
    ycbcr_to_srgb : Similar conversion with offset U/V (digital video).

    References
    ----------
    .. [1] ITU-R BT.601-7, "Studio encoding parameters of digital television
           for standard 4:3 and wide screen 16:9 aspect ratios"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"yuv_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.yuv_to_srgb(input)
