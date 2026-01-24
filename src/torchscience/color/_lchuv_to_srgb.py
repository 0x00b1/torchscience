"""LCHuv to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def lchuv_to_srgb(input: Tensor) -> Tensor:
    r"""Convert LCHuv to sRGB color space.

    Converts input colors from LCHuv (cylindrical CIELUV) to sRGB (with gamma).
    Uses D65 illuminant as the reference white point.

    Mathematical Definition
    -----------------------
    The conversion first transforms LCHuv to CIELUV:

    .. math::
        L^* = L^*

    .. math::
        u^* = C^* \cdot \cos(h)

    .. math::
        v^* = C^* \cdot \sin(h)

    Then converts CIELUV to sRGB via XYZ color space.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        LCHuv color values [L*, C*, h] where:
        - L* is lightness in [0, 100] for SDR content
        - C* is chroma (saturation), non-negative
        - h is hue angle in radians

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values. Values may be outside [0, 1] for out-of-gamut colors.

    Examples
    --------
    Convert LCHuv white to sRGB:

    >>> lch = torch.tensor([[100.0, 0.0, 0.0]])
    >>> torchscience.color.lchuv_to_srgb(lch)
    tensor([[1.0000, 1.0000, 1.0000]])

    Convert a saturated red-ish color:

    >>> import math
    >>> lch = torch.tensor([[50.0, 50.0, math.pi / 6]])  # h = 30 degrees
    >>> rgb = torchscience.color.lchuv_to_srgb(lch)

    See Also
    --------
    srgb_to_lchuv : Inverse conversion from sRGB to LCHuv.
    luv_to_srgb : Convert CIELUV (rectangular form) to sRGB.

    References
    ----------
    .. [1] CIE 15:2004, "Colorimetry, 3rd Edition"
    .. [2] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"lchuv_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.lchuv_to_srgb(input)
