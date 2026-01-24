"""YPbPr to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def ypbpr_to_srgb(input: Tensor) -> Tensor:
    r"""Convert YPbPr to sRGB color space (BT.601 inverse).

    Converts input colors from YPbPr to sRGB using the inverse BT.601
    transformation followed by gamma encoding.

    Mathematical Definition
    -----------------------
    First applies inverse BT.601 matrix, then gamma encodes:

    .. math::
        R_{lin} &= Y + 1.402 Pr \\
        G_{lin} &= Y - 0.344136 Pb - 0.714136 Pr \\
        B_{lin} &= Y + 1.772 Pb

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        YPbPr color values where Y is in [0, 1] and Pb, Pr are in [-0.5, 0.5].

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values in [0, 1].

    Examples
    --------
    Convert white from YPbPr to sRGB:

    >>> ypbpr = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.ypbpr_to_srgb(ypbpr)
    tensor([[1.0000, 1.0000, 1.0000]])

    See Also
    --------
    srgb_to_ypbpr : Forward conversion from sRGB to YPbPr.
    ycbcr_to_srgb : Similar conversion but for YCbCr with offset.

    References
    ----------
    .. [1] ITU-R BT.601-7, "Studio encoding parameters of digital television
           for standard 4:3 and wide screen 16:9 aspect ratios"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"ypbpr_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.ypbpr_to_srgb(input)
