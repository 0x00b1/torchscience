"""sRGB to CIELUV color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_luv(input: Tensor) -> Tensor:
    r"""Convert sRGB to CIELUV color space.

    Converts input colors from sRGB (with gamma) to CIELUV (L*u*v*) color space.
    Uses D65 illuminant as the reference white point.

    Mathematical Definition
    -----------------------
    The conversion first linearizes sRGB and converts to XYZ, then applies
    the LUV transform:

    .. math::
        L^* = \begin{cases}
            116 \cdot (Y/Y_n)^{1/3} - 16 & \text{if } Y/Y_n > \delta^3 \\
            (29/3)^3 \cdot Y/Y_n & \text{otherwise}
        \end{cases}

    where :math:`\delta = 6/29` and :math:`Y_n = 1.0` is the D65 white point Y.

    The chromaticity coordinates are computed as:

    .. math::
        u' = \frac{4X}{X + 15Y + 3Z}, \quad v' = \frac{9Y}{X + 15Y + 3Z}

    And the u* and v* components are:

    .. math::
        u^* = 13 L^* (u' - u'_n), \quad v^* = 13 L^* (v' - v'_n)

    where :math:`u'_n \approx 0.19784` and :math:`v'_n \approx 0.46834` are the
    D65 white point chromaticity coordinates.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. Values outside [0, 1] are allowed (HDR).

    Returns
    -------
    Tensor, shape (..., 3)
        CIELUV color values [L*, u*, v*] where:
        - L* is lightness in [0, 100] for SDR content
        - u* is typically in [-100, 100]
        - v* is typically in [-100, 100]

    Examples
    --------
    Convert D65 white to LUV (should give L=100, u=0, v=0):

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_luv(rgb)
    tensor([[100.0000,   0.0000,   0.0000]])

    Convert pure red:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> luv = torchscience.color.srgb_to_luv(rgb)
    >>> luv[:, 0]  # L* component
    tensor([53.2329])

    See Also
    --------
    luv_to_srgb : Inverse conversion from CIELUV to sRGB.
    srgb_to_xyz : Convert sRGB to XYZ (intermediate space).
    srgb_to_lab : Convert sRGB to CIELAB (similar perceptual space).

    References
    ----------
    .. [1] CIE 15:2004, "Colorimetry, 3rd Edition"
    .. [2] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_luv: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_luv(input)
