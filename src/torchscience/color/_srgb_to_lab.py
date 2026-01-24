"""sRGB to CIELAB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_lab(input: Tensor) -> Tensor:
    r"""Convert sRGB to CIELAB color space.

    Converts input colors from sRGB (with gamma) to CIELAB (L*a*b*) color space.
    Uses D65 illuminant as the reference white point.

    Mathematical Definition
    -----------------------
    The conversion first linearizes sRGB and converts to XYZ, then applies
    the Lab transform:

    .. math::
        L^* = 116 \cdot f(Y/Y_n) - 16

    .. math::
        a^* = 500 \cdot (f(X/X_n) - f(Y/Y_n))

    .. math::
        b^* = 200 \cdot (f(Y/Y_n) - f(Z/Z_n))

    where :math:`(X_n, Y_n, Z_n) = (0.95047, 1.0, 1.08883)` is the D65 white
    point and:

    .. math::
        f(t) = \begin{cases}
            t^{1/3} & \text{if } t > \delta^3 \\
            \frac{t}{3\delta^2} + \frac{4}{29} & \text{otherwise}
        \end{cases}

    with :math:`\delta = 6/29`.

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. Values outside [0, 1] are allowed (HDR).

    Returns
    -------
    Tensor, shape (..., 3)
        CIELAB color values [L*, a*, b*] where:
        - L* is lightness in [0, 100] for SDR content
        - a* is green-red axis (typically [-128, 127])
        - b* is blue-yellow axis (typically [-128, 127])

    Examples
    --------
    Convert D65 white to Lab (should give L=100, a=0, b=0):

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_lab(rgb)
    tensor([[100.0000,   0.0000,   0.0000]])

    Convert pure red:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> lab = torchscience.color.srgb_to_lab(rgb)
    >>> lab[:, 0]  # L* component
    tensor([53.2329])

    See Also
    --------
    lab_to_srgb : Inverse conversion from CIELAB to sRGB.
    srgb_to_xyz : Convert sRGB to XYZ (intermediate space).

    References
    ----------
    .. [1] CIE 15:2004, "Colorimetry, 3rd Edition"
    .. [2] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management -
           Default RGB colour space - sRGB"
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_lab: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_lab(input)
