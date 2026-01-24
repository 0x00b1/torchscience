"""YCbCr to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def ycbcr_to_srgb(input: Tensor) -> Tensor:
    r"""Convert YCbCr to sRGB color space (ITU-R BT.601).

    Converts input colors from YCbCr to sRGB using the inverse BT.601
    transformation.

    Mathematical Definition
    -----------------------
    The inverse conversion is:

    .. math::
        R &= Y + 1.402 (Cr - 0.5) \\
        G &= Y - 0.344136 (Cb - 0.5) - 0.714136 (Cr - 0.5) \\
        B &= Y + 1.772 (Cb - 0.5)

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        YCbCr color values where Y is in [0, 1] and Cb, Cr are in [0, 1]
        (centered at 0.5).

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values in [0, 1].

    Examples
    --------
    Convert white from YCbCr to sRGB:

    >>> ycbcr = torch.tensor([[1.0, 0.5, 0.5]])
    >>> torchscience.color.ycbcr_to_srgb(ycbcr)
    tensor([[1.0000, 1.0000, 1.0000]])

    See Also
    --------
    srgb_to_ycbcr : Forward conversion from sRGB to YCbCr.

    References
    ----------
    .. [1] ITU-R BT.601-7, "Studio encoding parameters of digital television
           for standard 4:3 and wide screen 16:9 aspect ratios"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"ycbcr_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.ycbcr_to_srgb(input)
