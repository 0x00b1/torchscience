"""IntersectionResult tensorclass for intersection results."""

from __future__ import annotations

from tensordict import tensorclass
from torch import Tensor


@tensorclass
class IntersectionResult:
    """Ray-geometry intersection result for primitive shapes.

    Attributes
    ----------
    t : Tensor, shape (*,)
        Hit distance along ray. inf if miss.
    hit_point : Tensor, shape (*, 3)
        World-space intersection point. (0,0,0) if miss.
    normal : Tensor, shape (*, 3)
        Surface normal at hit point (normalized). (0,0,0) if miss.
    uv : Tensor, shape (*, 2)
        Parametric coordinates (primitive-specific). (0,0) if miss.
    hit : Tensor, shape (*,)
        Boolean mask, True if valid hit.
    """

    t: Tensor
    hit_point: Tensor
    normal: Tensor
    uv: Tensor
    hit: Tensor
