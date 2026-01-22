"""Mathematical morphology closing operation."""

from typing import Sequence

from torch import Tensor

from torchscience.morphology._dilation import dilation
from torchscience.morphology._erosion import erosion


def closing(
    input: Tensor,
    structuring_element: Tensor,
    *,
    origin: Sequence[int] | None = None,
    padding_mode: str = "zeros",
) -> Tensor:
    r"""Compute N-dimensional morphological closing.

    Closing is dilation followed by erosion with the same structuring element.
    It fills small dark holes and smooths object boundaries while
    approximately preserving object sizes and shapes.

    Mathematical Definition
    -----------------------
    .. math::
        \varphi_B(f) = \varepsilon_B(\delta_B(f))

    where :math:`\delta_B` is dilation and :math:`\varepsilon_B` is erosion.

    Parameters
    ----------
    input : Tensor, shape (*, spatial...)
        Input tensor. The trailing dimensions are treated as spatial dimensions
        matching the structuring element's dimensions.
    structuring_element : Tensor
        Structuring element. If dtype is bool, flat morphology is performed.
        If dtype is float, grayscale (non-flat) morphology is performed.
    origin : Sequence[int], optional
        Origin (anchor point) of the structuring element. Default is the center.
    padding_mode : str, optional
        How to handle boundaries. One of ``"zeros"``, ``"reflect"``,
        ``"replicate"``, ``"circular"``. Default: ``"zeros"``.

    Returns
    -------
    Tensor
        Closed tensor with the same shape as input.

    Examples
    --------
    Fill small dark holes:

    >>> image = torch.rand(64, 64)
    >>> se = torch.ones(3, 3, dtype=torch.bool)
    >>> closed = torchscience.morphology.closing(image, se)

    Notes
    -----
    - Closing is idempotent: ``closing(closing(f, B), B) = closing(f, B)``.
    - Closing is extensive: ``closing(f, B) >= f``.
    - Closing preserves relative ordering: if ``f <= g``, then
      ``closing(f, B) <= closing(g, B)``.
    - Closing is the dual of opening: ``closing(f, B) = -opening(-f, B)``.

    See Also
    --------
    opening : Dual operation (erosion followed by dilation).
    dilation : First step of closing.
    erosion : Second step of closing.

    References
    ----------
    .. [1] J. Serra, "Image Analysis and Mathematical Morphology", Academic Press, 1982.
    """
    dilated = dilation(
        input,
        structuring_element,
        origin=origin,
        padding_mode=padding_mode,
    )
    return erosion(
        dilated,
        structuring_element,
        origin=origin,
        padding_mode=padding_mode,
    )
