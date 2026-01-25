"""Spatial data structures for efficient neighbor search and range queries.

This module provides k-d trees, bounding volume hierarchies (BVH), and octrees with:
- O(log n) query performance via tree traversal
- Autograd support for query operations (build is non-differentiable)
- Batched construction for (B, N, D) point clouds
- Thread-safe concurrent queries

Note: Tree construction produces a discrete data structure and is NOT
differentiable. Query operations (k_nearest_neighbors, range_search,
octree_sample) support autograd through the returned data/distances.
"""

from ._bounding_volume_hierarchy import (
    BoundingVolumeHierarchy,
    bounding_volume_hierarchy,
)
from ._k_nearest_neighbors import k_nearest_neighbors
from ._kd_tree import KdTree, kd_tree
from ._octree import Octree, octree
from ._octree_dynamic import (
    octree_insert,
    octree_merge,
    octree_remove,
    octree_subdivide,
)
from ._octree_neighbors import octree_neighbors
from ._octree_ray_marching import octree_ray_marching
from ._octree_sample import octree_sample
from ._octree_structure_learning import (
    octree_adaptive_subdivide,
    octree_subdivision_scores,
)
from ._range_search import range_search

__all__ = [
    "BoundingVolumeHierarchy",
    "KdTree",
    "Octree",
    "bounding_volume_hierarchy",
    "k_nearest_neighbors",
    "kd_tree",
    "octree",
    "octree_adaptive_subdivide",
    "octree_insert",
    "octree_merge",
    "octree_neighbors",
    "octree_ray_marching",
    "octree_remove",
    "octree_sample",
    "octree_subdivide",
    "octree_subdivision_scores",
    "range_search",
]
