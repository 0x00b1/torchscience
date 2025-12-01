"""Reinhard tone mapping implementation."""

from typing import Optional, Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def reinhard(
    input: Tensor,
    *,
    white_point: Optional[Union[Tensor, float]] = None,
) -> Tensor:
    r"""Apply Reinhard tone mapping.

    Maps HDR (High Dynamic Range) values to LDR (Low Dynamic Range) [0, 1]
    using the Reinhard operator.

    Mathematical Definition
    -----------------------
    Basic Reinhard (no white point):

    .. math::
        L_{out} = \frac{L_{in}}{1 + L_{in}}

    Extended Reinhard (with white point):

    .. math::
        L_{out} = \frac{L_{in} (1 + L_{in}/L_{white}^2)}{1 + L_{in}}

    Parameters
    ----------
    input : Tensor
        Input HDR values. Can be any shape.
    white_point : Tensor or float, optional
        The luminance value that maps to white (1.0). If provided, uses the
        extended Reinhard operator which ensures that white_point maps exactly
        to 1.0. If None, uses the basic Reinhard operator.

    Returns
    -------
    Tensor
        Tone-mapped values in [0, 1] (or [0, 1] with extended).

    Examples
    --------
    Basic Reinhard:

    >>> hdr = torch.tensor([0.0, 1.0, 10.0, 100.0])
    >>> torchscience.graphics.tone_mapping.reinhard(hdr)
    tensor([0.0000, 0.5000, 0.9091, 0.9901])

    Extended Reinhard with white point:

    >>> hdr = torch.tensor([0.0, 5.0, 10.0])
    >>> torchscience.graphics.tone_mapping.reinhard(hdr, white_point=10.0)
    tensor([0.0000, 0.5833, 1.0000])

    Notes
    -----
    - Basic Reinhard asymptotically approaches 1 but never reaches it.
    - Extended Reinhard with white_point ensures that luminance values at
      white_point map exactly to 1.0.
    - Values above white_point will map to values above 1.0.

    References
    ----------
    .. [1] E. Reinhard et al., "Photographic Tone Reproduction for Digital
           Images", ACM SIGGRAPH 2002.
    """
    if white_point is not None and not isinstance(white_point, Tensor):
        white_point = torch.tensor(
            white_point, device=input.device, dtype=input.dtype
        )

    return torch.ops.torchscience.reinhard(input, white_point)
