"""sRGB to LCHuv color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_lchuv(input: Tensor) -> Tensor:
    r"""Convert sRGB to LCHuv color space.

    Converts input colors from sRGB (with gamma) to LCHuv, the cylindrical
    representation of CIELUV. Uses D65 illuminant as the reference white point.

    Mathematical Definition
    -----------------------
    LCHuv is CIELUV expressed in cylindrical coordinates:

    .. math::
        L^* = L^* \text{ (same as LUV)}

    .. math::
        C^* = \sqrt{u^{*2} + v^{*2}}

    .. math::
        h = \arctan2(v^*, u^*)

    where :math:`C^*` is the chroma (colorfulness) and :math:`h` is the hue angle
    in radians.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. Values outside [0, 1] are allowed (HDR).

    Returns
    -------
    Tensor, shape (..., 3)
        LCHuv color values [L*, C*, h] where:
        - L* is lightness in [0, 100] for SDR content
        - C* is chroma (saturation), non-negative
        - h is hue angle in radians [-pi, pi]

    Examples
    --------
    Convert D65 white to LCHuv (should give L=100, C=0, h=0):

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_lchuv(rgb)
    tensor([[100.0000,   0.0000,   0.0000]])

    Convert pure red:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> lch = torchscience.color.srgb_to_lchuv(rgb)
    >>> lch[:, 0]  # L* component
    tensor([53.2329])

    See Also
    --------
    lchuv_to_srgb : Inverse conversion from LCHuv to sRGB.
    srgb_to_luv : Convert sRGB to CIELUV (rectangular form).

    References
    ----------
    .. [1] CIE 15:2004, "Colorimetry, 3rd Edition"
    .. [2] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_lchuv: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_lchuv(input)
