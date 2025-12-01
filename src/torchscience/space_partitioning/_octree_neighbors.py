"""Octree neighbor finding operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._octree import Octree


def octree_neighbors(
    tree: "Octree",
    query_codes: Tensor,
    *,
    connectivity: Literal[6, 18, 26] = 6,
) -> tuple[Tensor, Tensor]:
    """Find neighbors for voxels in the octree.

    Parameters
    ----------
    tree : Octree
        Sparse voxel structure with full hierarchy.
    query_codes : Tensor, shape (n_queries,)
        Morton codes of voxels to find neighbors for. Must be codes that exist
        in ``tree.codes``.
    connectivity : {6, 18, 26}, default=6
        Neighborhood connectivity:
        - 6: Face neighbors only (±x, ±y, ±z)
        - 18: Face + edge neighbors
        - 26: Face + edge + corner neighbors (full 3D neighborhood)

    Returns
    -------
    neighbor_codes : Tensor, shape (n_queries, connectivity)
        Morton codes of neighbor voxels. Value is -1 if no neighbor exists
        (boundary or empty region).
    neighbor_data : Tensor, shape (n_queries, connectivity, *value_shape)
        Data of neighbor voxels. Zero-filled for non-existent neighbors.

    Notes
    -----
    **LOD-aware neighbor finding:** For a voxel at depth D, neighbors are found
    using top-down traversal:

    1. Compute the neighbor's expected center position (voxel center + offset)
    2. Check if position is inside [-1, 1]^3 (boundary check)
    3. Traverse from root to depth D using octant computation
    4. If the neighbor exists at the same depth, return it
    5. If traversal ends at a coarser leaf (depth < D), return that ancestor
    6. If traversal enters an empty region (no child), return -1

    **Fixed-connectivity output:** This function returns fixed-size outputs per
    query, enabling torch.compile compatibility. Use ``neighbor_codes == -1``
    to identify missing neighbors.

    **Mixed-depth handling:** At LOD boundaries where a fine voxel is adjacent
    to a coarse voxel, the neighbor query returns the coarse ancestor. This is
    consistent with sparse octree representations where merged regions are
    represented by a single coarse node.

    **Neighbor offset directions:**

    Face neighbors (connectivity 6):
        - Index 0-5: -x, +x, -y, +y, -z, +z

    Edge neighbors (connectivity 18, indices 6-17):
        - Index 6-9: xy edges (-x-y, -x+y, +x-y, +x+y, z=0)
        - Index 10-13: xz edges (-x-z, -x+z, +x-z, +x+z, y=0)
        - Index 14-17: yz edges (-y-z, -y+z, +y-z, +y+z, x=0)

    Corner neighbors (connectivity 26, indices 18-25):
        - Index 18-25: all 8 corners (±x, ±y, ±z combinations)

    Examples
    --------
    >>> tree = octree(points, features, maximum_depth=8)
    >>> # Find face neighbors for all leaf voxels
    >>> leaf_codes = tree.codes[tree.children_mask == 0]
    >>> neighbor_codes, neighbor_data = octree_neighbors(
    ...     tree, leaf_codes, connectivity=6
    ... )
    >>> # Check which neighbors exist
    >>> has_neighbor = neighbor_codes != -1
    """
    if connectivity not in (6, 18, 26):
        raise ValueError(
            f"connectivity must be 6, 18, or 26, got {connectivity}"
        )

    # Call C++ kernel
    return torch.ops.torchscience.octree_neighbors(
        tree.data,
        tree.codes,
        tree.structure,
        tree.children_mask,
        query_codes,
        connectivity,
    )
