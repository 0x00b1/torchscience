"""Edge gradient contribution for differentiable rendering."""

from __future__ import annotations

import torch
from torch import Tensor

from ._edge_sampling_result import EdgeSamples


def edge_gradient_contribution(
    edge_samples: EdgeSamples,
    delta_intensity: Tensor,
    ray_directions: Tensor,
    screen_normals: Tensor | None = None,
) -> Tensor:
    r"""Compute the gradient contribution from silhouette edge samples.

    Evaluates the edge-based Monte Carlo estimator for the boundary integral
    in differentiable rendering. Each sample point on a silhouette edge
    contributes a gradient term proportional to the intensity difference
    across the edge, weighted by the geometric factor
    :math:`\ell_e / (n_e \cdot d)`.

    Mathematical Definition
    -----------------------
    For an edge sample with tangent :math:`\hat{e}`, ray direction :math:`d`,
    edge length :math:`\ell_e`, and intensity difference
    :math:`\Delta I`, the contribution is:

    .. math::
        G = \Delta I \cdot \frac{\ell_e}{n_e \cdot d}

    where the edge normal :math:`n_e` is computed as the normalized rejection
    of :math:`d` from :math:`\hat{e}`:

    .. math::
        n_e = \frac{d - (d \cdot \hat{e})\,\hat{e}}
              {\lVert d - (d \cdot \hat{e})\,\hat{e} \rVert}

    Parameters
    ----------
    edge_samples : EdgeSamples
        Samples along silhouette edges. Must contain ``edge_tangent`` of
        shape ``(N, 3)`` and ``edge_length`` of shape ``(N,)``.
    delta_intensity : Tensor, shape (N,) or (N, C)
        Intensity difference across the edge at each sample point. Scalar
        ``(N,)`` for single-channel or ``(N, C)`` for multi-channel (e.g.
        RGB with ``C=3``).
    ray_directions : Tensor, shape (N, 3)
        Direction vectors of the rays through each sample point.
    screen_normals : Tensor or None, shape (N, 3), optional
        If provided, used directly as the edge normal in screen space.
        Otherwise the edge normal is computed from the ray direction and
        edge tangent via vector rejection.

    Returns
    -------
    Tensor, shape (N,) or (N, C)
        Gradient contribution for each edge sample, matching the shape of
        ``delta_intensity``.

    Raises
    ------
    ValueError
        If ``ray_directions`` does not have shape ``(N, 3)``.
    ValueError
        If ``delta_intensity`` first dimension does not match the number
        of edge samples.
    ValueError
        If ``screen_normals`` is provided but does not have shape ``(N, 3)``.
    """
    e_hat = edge_samples.edge_tangent  # (N, 3)
    edge_length = edge_samples.edge_length  # (N,)
    n_samples = e_hat.shape[0]

    # --- input validation ---------------------------------------------------
    if ray_directions.ndim != 2 or ray_directions.shape[1] != 3:
        raise ValueError(
            f"ray_directions must have shape (N, 3), got {ray_directions.shape}"
        )
    if ray_directions.shape[0] != n_samples:
        raise ValueError(
            f"ray_directions has {ray_directions.shape[0]} samples but "
            f"edge_samples has {n_samples}"
        )
    if delta_intensity.shape[0] != n_samples:
        raise ValueError(
            f"delta_intensity first dimension ({delta_intensity.shape[0]}) "
            f"must match number of edge samples ({n_samples})"
        )
    if screen_normals is not None:
        if screen_normals.ndim != 2 or screen_normals.shape != (n_samples, 3):
            raise ValueError(
                f"screen_normals must have shape ({n_samples}, 3), "
                f"got {screen_normals.shape}"
            )

    # --- edge normal computation ---------------------------------------------
    if screen_normals is not None:
        n_e = screen_normals
    else:
        # Rejection of ray direction from tangent: d - (d . e_hat) * e_hat
        d_dot_e = (ray_directions * e_hat).sum(dim=-1, keepdim=True)  # (N, 1)
        rejection = ray_directions - d_dot_e * e_hat  # (N, 3)
        n_e = torch.nn.functional.normalize(rejection, dim=-1)  # (N, 3)

    # --- geometric weight ----------------------------------------------------
    n_dot_d = (n_e * ray_directions).sum(dim=-1)  # (N,)
    n_dot_d = n_dot_d.clamp(min=1e-8)  # avoid division by zero

    weight = edge_length / n_dot_d  # (N,)

    # --- multiply by delta_intensity -----------------------------------------
    if delta_intensity.ndim == 2:
        # Multi-channel: broadcast weight (N,) -> (N, 1)
        weight = weight.unsqueeze(-1)

    return delta_intensity * weight
