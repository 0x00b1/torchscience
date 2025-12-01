"""Linear sRGB to sRGB color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_linear_to_srgb(input: Tensor) -> Tensor:
    r"""Convert linear sRGB to sRGB color space.

    Converts input colors from linear sRGB to sRGB (gamma-encoded) color space
    using the IEC 61966-2-1 standard inverse transfer function. The conversion is
    differentiable and supports arbitrary tensor shapes.

    Mathematical Definition
    -----------------------
    For each input value :math:`x`:

    .. math::
        f(x) = \begin{cases}
            12.92 \cdot x & \text{if } x \leq 0.0031308 \\
            1.055 \cdot x^{1/2.4} - 0.055 & \text{otherwise}
        \end{cases}

    The backward gradient is:

    .. math::
        \frac{\partial f}{\partial x} = \begin{cases}
            12.92 & \text{if } x \leq 0.0031308 \\
            \frac{1.055}{2.4} \cdot x^{1/2.4 - 1} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    input : Tensor
        Linear sRGB color values, typically in the range [0, 1]. Can be any shape,
        e.g., (3,) for a single RGB pixel, (H, W, 3) for an image, or
        (B, H, W, 3) for a batch of images.

    Returns
    -------
    Tensor
        sRGB (gamma-encoded) color values with the same shape as input.

    Examples
    --------
    Convert a single linear value:

    >>> linear = torch.tensor([0.214])
    >>> torchscience.graphics.color.srgb_linear_to_srgb(linear)
    tensor([0.5000])

    Convert an RGB triplet:

    >>> linear = torch.tensor([0.0331, 0.2140, 0.6038])
    >>> torchscience.graphics.color.srgb_linear_to_srgb(linear)
    tensor([0.2000, 0.5000, 0.8000])

    Convert values in the linear region (below threshold 0.0031308):

    >>> linear = torch.tensor([0.0008, 0.0015, 0.0031])
    >>> srgb = torchscience.graphics.color.srgb_linear_to_srgb(linear)
    >>> srgb  # srgb = linear * 12.92
    tensor([0.0103, 0.0194, 0.0401])

    Notes
    -----
    - The threshold value 0.0031308 is defined by IEC 61966-2-1.
    - Values below the threshold use a linear formula to avoid numerical
      issues near zero.
    - Gradients are computed analytically and support backpropagation.
    - The function is continuous at the threshold point.

    See Also
    --------
    srgb_to_srgb_linear : Inverse conversion from sRGB to linear sRGB.

    References
    ----------
    .. [1] IEC 61966-2-1:1999, "Multimedia systems and equipment - Colour
           measurement and management - Part 2-1: Colour management - Default
           RGB colour space - sRGB"
    """
    return torch.ops.torchscience.srgb_linear_to_srgb(input)
