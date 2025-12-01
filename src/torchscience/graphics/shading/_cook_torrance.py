"""Cook-Torrance BRDF implementation."""

from typing import Union

import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - Load C++ operators


def cook_torrance(
    normal: Tensor,
    view: Tensor,
    light: Tensor,
    *,
    roughness: Union[Tensor, float],
    f0: Union[Tensor, float] = 0.04,
) -> Tensor:
    r"""Compute Cook-Torrance specular BRDF.

    Evaluates the Cook-Torrance microfacet specular BRDF using GGX distribution,
    Schlick-GGX geometry, and Schlick Fresnel approximation.

    Mathematical Definition
    -----------------------
    The specular BRDF is:

    .. math::
        f_r = \frac{D \cdot F \cdot G}{4(n \cdot l)(n \cdot v)}

    Where:

    - :math:`D`: GGX/Trowbridge-Reitz normal distribution
    - :math:`F`: Schlick Fresnel approximation
    - :math:`G`: Schlick-GGX geometry with Smith masking-shadowing

    Parameters
    ----------
    normal : Tensor, shape (..., 3)
        Surface normal vectors. Must be normalized.
    view : Tensor, shape (..., 3)
        View direction vectors (toward camera). Must be normalized.
    light : Tensor, shape (..., 3)
        Light direction vectors (toward light). Must be normalized.
    roughness : Tensor or float, shape (...) or scalar
        Surface roughness in range [0, 1]. Values are clamped to [0.001, 1.0]
        to avoid singularities.
    f0 : Tensor or float, shape (...) or (..., 3) or scalar, default=0.04
        Fresnel reflectance at normal incidence. Use 0.04 for common
        dielectrics (plastic, glass). For metals, use the material's
        RGB reflectance values.

    Returns
    -------
    Tensor
        Specular BRDF values. Shape depends on inputs:

        - If f0 is scalar or shape (...): returns shape (...)
        - If f0 is shape (..., 3): returns shape (..., 3) for RGB

    Examples
    --------
    Basic dielectric surface (plastic-like):

    >>> normal = torch.tensor([[0.0, 1.0, 0.0]])  # Up
    >>> view = torch.tensor([[0.0, 0.707, 0.707]])  # 45 degrees
    >>> light = torch.tensor([[0.0, 0.707, -0.707]])  # 45 degrees opposite
    >>> torchscience.graphics.shading.cook_torrance(
    ...     normal, view, light, roughness=0.5
    ... )
    tensor([...])

    Metallic surface with RGB reflectance:

    >>> f0_gold = torch.tensor([1.0, 0.71, 0.29])  # Gold reflectance
    >>> torchscience.graphics.shading.cook_torrance(
    ...     normal, view, light, roughness=0.3, f0=f0_gold
    ... )
    tensor([[...]])

    Notes
    -----
    - All direction vectors (normal, view, light) must be normalized.
    - The function returns 0 when n·l <= 0 or n·v <= 0 (back-facing).
    - Roughness is internally clamped to [0.001, 1.0] for numerical stability.
    - This implementation follows Unreal Engine 4's approach from
      "Real Shading in Unreal Engine 4" (SIGGRAPH 2013).
    - **Gradient Support**: Full autograd support with first-order and
      second-order (Hessian) gradients. Use ``create_graph=True`` for
      second-order optimization in inverse rendering applications.

    References
    ----------
    .. [1] B. Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013.
           https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
    .. [2] LearnOpenGL, "PBR Theory",
           https://learnopengl.com/PBR/Theory
    """
    # Input validation
    if normal.shape[-1] != 3:
        raise ValueError(
            f"normal must have last dimension 3, got {normal.shape[-1]}"
        )
    if view.shape[-1] != 3:
        raise ValueError(
            f"view must have last dimension 3, got {view.shape[-1]}"
        )
    if light.shape[-1] != 3:
        raise ValueError(
            f"light must have last dimension 3, got {light.shape[-1]}"
        )

    # Convert scalars to tensors (use normal's device, dtype will be promoted later)
    if not isinstance(roughness, Tensor):
        roughness = torch.tensor(roughness, device=normal.device)
    if not isinstance(f0, Tensor):
        f0 = torch.tensor(f0, device=normal.device)

    # Ensure tensors are on the same device and have compatible dtypes
    # Include all tensor inputs in dtype promotion
    target_dtype = normal.dtype
    for t in [view, light, roughness, f0]:
        target_dtype = torch.promote_types(target_dtype, t.dtype)
    target_device = normal.device

    if normal.dtype != target_dtype:
        normal = normal.to(target_dtype)
    if view.dtype != target_dtype:
        view = view.to(target_dtype)
    if light.dtype != target_dtype:
        light = light.to(target_dtype)
    if roughness.dtype != target_dtype:
        roughness = roughness.to(target_dtype)
    if f0.dtype != target_dtype:
        f0 = f0.to(target_dtype)

    if roughness.device != target_device:
        roughness = roughness.to(target_device)
    if f0.device != target_device:
        f0 = f0.to(target_device)

    return torch.ops.torchscience.cook_torrance(
        normal, view, light, roughness, f0
    )
