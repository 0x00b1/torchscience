"""sRGB to Oklab color conversion."""

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def srgb_to_oklab(input: Tensor) -> Tensor:
    r"""Convert sRGB to Oklab color space.

    Converts input colors from sRGB (with gamma) to Oklab, a perceptually
    uniform color space designed by Bjorn Ottosson.

    Mathematical Definition
    -----------------------
    The conversion first linearizes sRGB, then applies two matrix transforms
    with a cube root nonlinearity:

    1. Linearize sRGB using the standard transfer function
    2. Apply M1 matrix to get LMS cone response:

    .. math::
        \begin{pmatrix} l \\ m \\ s \end{pmatrix} =
        M_1 \begin{pmatrix} R_{linear} \\ G_{linear} \\ B_{linear} \end{pmatrix}

    3. Apply cube root:

    .. math::
        l' = \sqrt[3]{l}, \quad m' = \sqrt[3]{m}, \quad s' = \sqrt[3]{s}

    4. Apply M2 matrix to get Oklab:

    .. math::
        \begin{pmatrix} L \\ a \\ b \end{pmatrix} =
        M_2 \begin{pmatrix} l' \\ m' \\ s' \end{pmatrix}

    Parameters
    ----------
    input : Tensor, shape (..., 3)
        sRGB color values. Values outside [0, 1] are allowed (HDR).

    Returns
    -------
    Tensor, shape (..., 3)
        Oklab color values [L, a, b] where:
        - L is lightness in [0, 1] for SDR content
        - a is green-red axis
        - b is blue-yellow axis

    Examples
    --------
    Convert white to Oklab (should give L=1, a=0, b=0):

    >>> rgb = torch.tensor([[1.0, 1.0, 1.0]])
    >>> torchscience.color.srgb_to_oklab(rgb)
    tensor([[1.0000, 0.0000, 0.0000]])

    Convert pure red:

    >>> rgb = torch.tensor([[1.0, 0.0, 0.0]])
    >>> oklab = torchscience.color.srgb_to_oklab(rgb)
    >>> oklab[:, 0]  # L component
    tensor([0.6280])

    See Also
    --------
    oklab_to_srgb : Inverse conversion from Oklab to sRGB.
    srgb_to_lab : Convert sRGB to CIELAB (different from Oklab).

    References
    ----------
    .. [1] Bjorn Ottosson, "A perceptual color space for image processing",
           https://bottosson.github.io/posts/oklab/
    """
    if input.shape[-1] != 3:
        raise ValueError(
            f"srgb_to_oklab: input must have last dimension 3, got {input.shape[-1]}"
        )

    return torch.ops.torchscience.srgb_to_oklab(input)
