"""sRGB to LCHab color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_lchab(input: Tensor) -> Tensor:
    r"""Convert sRGB to LCHab color space.

    Converts input colors from sRGB (with gamma) to LCHab, the cylindrical
    representation of CIELAB. Uses D65 illuminant as the reference white point.

    Mathematical Definition
    -----------------------
    LCHab is CIELAB expressed in cylindrical coordinates:

    .. math::
        L^* = L^* \text{ (same as Lab)}

    .. math::
        C^* = \sqrt{a^{*2} + b^{*2}}

    .. math::
        h = \arctan2(b^*, a^*)

    where :math:`C^*` is the chroma (colorfulness) and :math:`h` is the hue angle
    in radians.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. Values outside [0, 1] are allowed (HDR).

    Returns
    -------
    Tensor, shape (..., 3)
        LCHab color values [L*, C*, h] where:
        - L* is lightness in [0, 100] for SDR content
        - C* is chroma (saturation), non-negative
        - h is hue angle in radians [-pi, pi]

    Examples
    --------
    Convert D65 white to LCHab (should give L=100, C=0, h=0):

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_lchab(rgb)
    tensor([[100.0000,   0.0000,   0.0000]])

    Convert pure red:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> lch = torchscience.color.srgb_to_lchab(rgb)
    >>> lch[:, 0]  # L* component
    tensor([53.2329])

    See Also
    --------
    lchab_to_srgb : Inverse conversion from LCHab to sRGB.
    srgb_to_lab : Convert sRGB to CIELAB (rectangular form).

    References
    ----------
    .. [1] CIE 15:2004, "Colorimetry, 3rd Edition"
    .. [2] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_lchab: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_lchab(input)
