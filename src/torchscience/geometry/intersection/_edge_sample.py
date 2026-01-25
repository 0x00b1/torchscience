"""Edge point sampling for differentiable silhouette rendering."""

from __future__ import annotations

import torch
from torch import Tensor

from ._edge_sampling_result import EdgeSamples, SilhouetteEdges


def edge_sample(
    vertices: Tensor,
    silhouette: SilhouetteEdges,
    num_samples_per_edge: int = 1,
    stratified: bool = True,
    generator: torch.Generator | None = None,
) -> EdgeSamples:
    r"""Sample points along silhouette edges for differentiable rendering.

    Generates sample points on each silhouette edge by linearly interpolating
    between the two edge endpoint vertices. The interpolation preserves the
    autograd graph so that gradients flow back to ``vertices``.

    Mathematical Definition
    -----------------------
    For each silhouette edge :math:`(v_i, v_j)` and parametric value
    :math:`t \in [0, 1]`, the sample position is:

    .. math::
        p = (1 - t) \, v_i + t \, v_j

    The edge tangent is the unit vector along the edge:

    .. math::
        \hat{e} = \frac{v_j - v_i}{\|v_j - v_i\|}

    When stratified sampling is enabled, each edge is divided into
    :math:`K` equal strata and one sample is drawn uniformly from each
    stratum:

    .. math::
        t_k \sim \mathcal{U}\!\left(\frac{k}{K},\, \frac{k+1}{K}\right),
        \quad k = 0, \ldots, K-1

    Parameters
    ----------
    vertices : Tensor, shape (num_vertices, 3)
        Vertex positions of the mesh. Gradients will flow through this
        tensor via the linear interpolation.
    silhouette : SilhouetteEdges
        Silhouette edges detected from a given viewpoint.
    num_samples_per_edge : int, default=1
        Number of sample points to generate per silhouette edge. Must be
        at least 1.
    stratified : bool, default=True
        If ``True``, use stratified sampling where each sample falls in a
        distinct uniform stratum along the edge. If ``False``, draw
        samples uniformly from :math:`[0, 1]`.
    generator : torch.Generator or None, default=None
        Optional random number generator for reproducibility.

    Returns
    -------
    EdgeSamples
        Tensorclass containing:

        - **positions**: World-space 3D positions of sample points,
          shape ``(num_sil * num_samples_per_edge, 3)``.
        - **edge_indices**: Index of the parent silhouette edge for each
          sample, shape ``(num_sil * num_samples_per_edge,)``.
        - **parametric_t**: Parametric coordinate :math:`t` in
          :math:`[0, 1]`, shape ``(num_sil * num_samples_per_edge,)``.
        - **edge_tangent**: Unit tangent vector along the parent edge,
          shape ``(num_sil * num_samples_per_edge, 3)``.
        - **edge_length**: Length of the parent edge,
          shape ``(num_sil * num_samples_per_edge,)``.

    Raises
    ------
    ValueError
        If ``vertices`` does not have shape ``(num_vertices, 3)``.
    ValueError
        If ``num_samples_per_edge`` is less than 1.

    Examples
    --------
    >>> vertices = torch.tensor([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [0.5, 1.0, 0.0],
    ... ])
    >>> sil = SilhouetteEdges(
    ...     edge_indices=torch.tensor([0]),
    ...     edges=torch.tensor([[0, 1]]),
    ...     front_face=torch.tensor([0]),
    ...     back_face=torch.tensor([-1]),
    ...     batch_size=[1],
    ... )
    >>> samples = edge_sample(vertices, sil, num_samples_per_edge=4)
    >>> samples.positions.shape
    torch.Size([4, 3])

    Notes
    -----
    The linear interpolation ``(1-t)*v_i + t*v_j`` uses ``vertices``
    directly (not detached) so that ``torch.autograd.grad`` can compute
    gradients of the sample positions with respect to vertex positions.
    This is essential for differentiable silhouette rendering pipelines
    such as [1]_.

    References
    ----------
    .. [1] S. Li, Z. Luan, et al., "Differentiable Rendering with Reparameterized
           Volume Sampling", 2023.
    """
    # --- Input validation ---------------------------------------------------
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"vertices must have shape (num_vertices, 3), got {vertices.shape}"
        )

    if num_samples_per_edge < 1:
        raise ValueError(
            f"num_samples_per_edge must be >= 1, got {num_samples_per_edge}"
        )

    # --- Shorthand -----------------------------------------------------------
    edges = silhouette.edges  # (num_sil, 2)
    num_sil = edges.shape[0]
    K = num_samples_per_edge
    total_samples = num_sil * K
    device = vertices.device
    dtype = vertices.dtype

    # --- Look up edge endpoint positions (differentiable indexing) -----------
    vi = vertices[edges[:, 0]]  # (num_sil, 3)
    vj = vertices[edges[:, 1]]  # (num_sil, 3)

    # --- Edge geometry (shared across samples on the same edge) -------------
    edge_vec = vj - vi  # (num_sil, 3)
    edge_len = torch.linalg.norm(edge_vec, dim=-1)  # (num_sil,)
    edge_tang = edge_vec / edge_len.unsqueeze(-1)  # (num_sil, 3)

    # --- Generate parametric t values ----------------------------------------
    if stratified:
        # Stratified: t_k ~ U(k/K, (k+1)/K) for k = 0 .. K-1
        strata_lo = torch.arange(K, device=device, dtype=dtype) / K  # (K,)
        u = torch.rand(
            num_sil, K, device=device, dtype=dtype, generator=generator
        )
        t = strata_lo.unsqueeze(0) + u / K  # (num_sil, K)
    else:
        # Uniform: t ~ U(0, 1)
        t = torch.rand(
            num_sil, K, device=device, dtype=dtype, generator=generator
        )

    # --- Compute sample positions via linear interpolation ------------------
    # vi, vj: (num_sil, 3) -> (num_sil, 1, 3)
    # t: (num_sil, K) -> (num_sil, K, 1)
    t_3d = t.unsqueeze(-1)  # (num_sil, K, 1)
    positions = (1.0 - t_3d) * vi.unsqueeze(1) + t_3d * vj.unsqueeze(1)
    # positions: (num_sil, K, 3)

    # --- Build per-sample edge indices ---------------------------------------
    edge_idx = (
        torch.arange(num_sil, device=device).unsqueeze(1).expand(num_sil, K)
    )

    # --- Expand edge tangent and length to per-sample -----------------------
    edge_tang_exp = edge_tang.unsqueeze(1).expand(num_sil, K, 3)
    edge_len_exp = edge_len.unsqueeze(1).expand(num_sil, K)

    # --- Flatten to (total_samples, ...) ------------------------------------
    return EdgeSamples(
        positions=positions.reshape(total_samples, 3),
        edge_indices=edge_idx.reshape(total_samples),
        parametric_t=t.reshape(total_samples),
        edge_tangent=edge_tang_exp.reshape(total_samples, 3),
        edge_length=edge_len_exp.reshape(total_samples),
        batch_size=[total_samples],
    )
