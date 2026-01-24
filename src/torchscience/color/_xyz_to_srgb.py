"""XYZ to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def xyz_to_srgb(input: Tensor) -> Tensor:
    r"""Convert CIE XYZ to sRGB color space.

    Converts input colors from CIE 1931 XYZ to sRGB (with gamma).
    Uses D65 illuminant and sRGB primaries per IEC 61966-2-1.

    Mathematical Definition
    -----------------------
    The conversion first applies the XYZ to linear RGB matrix:

    .. math::
        \begin{bmatrix} R \\ G \\ B \end{bmatrix}_{\text{linear}} =
        \begin{bmatrix}
             3.2404542 & -1.5371385 & -0.4985314 \\
            -0.9692660 &  1.8760108 &  0.0415560 \\
             0.0556434 & -0.2040259 &  1.0572252
        \end{bmatrix}
        \begin{bmatrix} X \\ Y \\ Z \end{bmatrix}

    Then applies gamma encoding:

    .. math::
        C_{\text{sRGB}} = \begin{cases}
            12.92 \times C & \text{if } C \leq 0.0031308 \\
            1.055 \times C^{1/2.4} - 0.055 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        CIE XYZ color values.

    Returns
    -------
    Tensor, shape (..., 3)
        sRGB color values. Values may be outside [0, 1] for out-of-gamut colors.

    Examples
    --------
    Convert D65 white from XYZ to sRGB:

    >>> xyz = torch.tensor([[0.9505, 1.0, 1.0890]])
    >>> torchscience.color.xyz_to_srgb(xyz)
    tensor([[1.0000, 1.0000, 1.0000]])

    See Also
    --------
    srgb_to_xyz : Inverse conversion from sRGB to XYZ.

    References
    ----------
    .. [1] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"xyz_to_srgb: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.xyz_to_srgb(input)
