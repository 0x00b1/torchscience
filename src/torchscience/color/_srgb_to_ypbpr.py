"""sRGB to YPbPr color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_ypbpr(input: Tensor) -> Tensor:
    r"""Convert sRGB to YPbPr color space (BT.601 analog component).

    Converts input colors from sRGB to YPbPr, the analog component video
    signal format used in component video cables. Unlike YCbCr which operates
    on gamma-encoded sRGB, YPbPr operates on linear (gamma-decoded) RGB.

    Mathematical Definition
    -----------------------
    First linearizes sRGB (gamma decode), then applies BT.601 matrix:

    .. math::
        Y  &= 0.299 R_{lin} + 0.587 G_{lin} + 0.114 B_{lin} \\
        Pb &= -0.168736 R_{lin} - 0.331264 G_{lin} + 0.5 B_{lin} \\
        Pr &= 0.5 R_{lin} - 0.418688 G_{lin} - 0.081312 B_{lin}

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values in [0, 1].

    Returns
    -------
    Tensor, shape (..., 3)
        YPbPr color values where Y is in [0, 1] and Pb, Pr are in [-0.5, 0.5].

    Examples
    --------
    Convert white to YPbPr:

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_ypbpr(rgb)
    tensor([[1.0000, 0.0000, 0.0000]])

    Convert red to YPbPr:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> torchscience.color.srgb_to_ypbpr(rgb)
    tensor([[0.2990, -0.1687,  0.5000]])

    See Also
    --------
    ypbpr_to_srgb : Inverse conversion from YPbPr to sRGB.
    srgb_to_ycbcr : Similar conversion but on gamma-encoded sRGB with offset.

    References
    ----------
    .. [1] ITU-R BT.601-7, "Studio encoding parameters of digital television
           for standard 4:3 and wide screen 16:9 aspect ratios"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_ypbpr: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_ypbpr(input)
