"""Test fixtures for space_partitioning tests."""

import pytest
import torch

from torchscience.geometry.space_partitioning import Octree


def _hash_code(code: int, capacity: int) -> int:
    """MurmurHash3 finalizer matching kernel implementation.

    IMPORTANT: Uses unsigned 64-bit arithmetic throughout to match C++ kernel.
    """
    h = code & 0xFFFFFFFFFFFFFFFF  # Treat as unsigned
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h &= 0xFFFFFFFFFFFFFFFF
    h ^= h >> 33
    return h & (capacity - 1)


def _decode_depth(code: int) -> int:
    """Extract depth from octree code using correct masking.

    CRITICAL: Must mask to 4 bits to handle depth >= 8 correctly.
    """
    return (code >> 60) & 0xF


def _hash_lookup(
    structure: torch.Tensor,
    codes: torch.Tensor,
    query_code: int,
    max_probes: int = 64,
) -> int:
    """Look up a code in the hash table with bounded linear probing.

    Returns the index in codes/data if found, -1 if not found.
    Used in tests to verify hash table correctness.
    """
    capacity = structure.shape[0]
    slot = _hash_code(query_code, capacity)

    for i in range(max_probes):
        idx = structure[(slot + i) % capacity].item()
        if idx == -1:
            return -1  # Empty slot, not found
        if codes[idx].item() == query_code:
            return idx
    return -1  # Exceeded max probes


def _code_to_center(code: int, depth: int) -> torch.Tensor:
    """Convert octree code to voxel center coordinates in [-1, 1]³.

    Used in tests to convert Morton codes back to spatial positions.
    """
    # Extract Morton part (mask off depth)
    morton = code & 0x0FFFFFFFFFFFFFFF

    # Decode Morton to x, y, z
    def compact_bits_3d(x):
        x = x & 0x1249249249249249
        x = (x | (x >> 2)) & 0x10C30C30C30C30C3
        x = (x | (x >> 4)) & 0x100F00F00F00F00F
        x = (x | (x >> 8)) & 0x1F0000FF0000FF
        x = (x | (x >> 16)) & 0x1F00000000FFFF
        x = (x | (x >> 32)) & 0x1FFFFF
        return x

    ix = compact_bits_3d(morton)
    iy = compact_bits_3d(morton >> 1)
    iz = compact_bits_3d(morton >> 2)

    # Convert to center coordinates in [-1, 1]³
    scale = 1 << depth
    voxel_size = 2.0 / scale
    x = -1.0 + (ix + 0.5) * voxel_size
    y = -1.0 + (iy + 0.5) * voxel_size
    z = -1.0 + (iz + 0.5) * voxel_size

    return torch.tensor([x, y, z])


def make_octree(
    codes: torch.Tensor,
    data: torch.Tensor,
    children_mask: torch.Tensor,
    weights: torch.Tensor,
    maximum_depth: int,
    capacity_factor: float = 2.0,
    aggregation: int = 0,
) -> Octree:
    """Test helper to construct Octree with explicit multi-depth structure.

    This bypasses the normal constructor to create specific LOD test cases.

    Parameters
    ----------
    codes : Tensor, shape (n,)
        Morton codes (with depth in bits 60-63).
    data : Tensor, shape (n, *value_shape)
        Voxel data.
    children_mask : Tensor, shape (n,), dtype=uint8
        Child existence bitmask per voxel.
    weights : Tensor, shape (n,)
        Point count per voxel.
    maximum_depth : int
        Maximum tree depth.
    capacity_factor : float
        Hash table size multiplier.
    aggregation : int
        Aggregation mode (0=mean, 1=sum, 2=max).

    Examples
    --------
    >>> # Create a tree with one coarse voxel and some fine children
    >>> codes = torch.tensor([
    ...     (4 << 60) | 0,      # depth 4, origin (internal node)
    ...     (8 << 60) | 0,      # depth 8, child at origin (leaf)
    ...     (8 << 60) | 1,      # depth 8, adjacent child (leaf)
    ... ])
    >>> data = torch.tensor([[1.0], [2.0], [3.0]])
    >>> children_mask = torch.tensor([0b00000011, 0, 0], dtype=torch.uint8)
    >>> weights = torch.tensor([2.0, 1.0, 1.0])
    >>> tree = make_octree(codes, data, children_mask, weights, maximum_depth=8)
    """
    count = codes.shape[0]
    capacity = int(count * capacity_factor)
    # Round up to power of 2
    if capacity < 1:
        capacity = 1
    capacity = 1 << (capacity - 1).bit_length()

    # Build hash table
    structure = torch.full((capacity,), -1, dtype=torch.int64)
    for i, code in enumerate(codes):
        slot = _hash_code(code.item(), capacity)
        while structure[slot] != -1:
            slot = (slot + 1) % capacity
        structure[slot] = i

    return Octree(
        codes=codes,
        data=data,
        structure=structure,
        children_mask=children_mask,
        weights=weights,
        maximum_depth=torch.tensor(maximum_depth, dtype=torch.int64),
        count=torch.tensor(count, dtype=torch.int64),
        aggregation=torch.tensor(aggregation, dtype=torch.int64),
        batch_size=[],
    )


@pytest.fixture
def simple_hierarchy_tree():
    """Fixture: octree with FULL hierarchy from depth 0 to depth 4.

    This fixture includes ALL intermediate nodes to satisfy the hierarchy
    invariant (all ancestors of every leaf exist).

    **Coordinate mapping:** Morton=0 at each depth corresponds to the min corner
    region near (-1, -1, -1) in normalized space, NOT the geometric origin.
    Queries into this tree should use coordinates like [-0.99, -0.99, -0.99].

    Structure:
    - Depth 0: root (1 node)
    - Depth 1: 1 internal node (child 0 of root = min corner octant)
    - Depth 2: 1 internal node (child 0 of depth-1)
    - Depth 3: 1 internal node (child 0 of depth-2)
    - Depth 4: 8 leaves (all children of depth-3 node)

    The 8 leaves at depth 4 cover the sub-octants of the min corner region.
    Their centers range from approximately (-0.9375, -0.9375, -0.9375) to
    (-0.8125, -0.8125, -0.8125) at depth 4 with voxel_size = 2/16 = 0.125.

    Total: 12 nodes (4 internal + 8 leaves)
    """
    codes = []
    children_masks = []
    data_values = []
    weights_values = []

    # Build path from root to depth 3 (all pointing to child 0 = min corner)
    for depth in range(4):
        code = depth << 60  # Morton part is 0 for child-0 path
        codes.append(code)
        children_masks.append(
            0x01 if depth < 3 else 0xFF
        )  # depth 3 has all 8 children
        data_values.append([3.5])  # Aggregated mean of leaves 0-7
        weights_values.append(8.0)

    # 8 leaves at depth 4 (children of the depth-3 node)
    for octant in range(8):
        # Child morton = (parent_morton << 3) | octant = 0 | octant = octant
        code = (4 << 60) | octant
        codes.append(code)
        children_masks.append(0)  # Leaf
        data_values.append([float(octant)])
        weights_values.append(1.0)

    codes_tensor = torch.tensor(codes, dtype=torch.int64)
    data_tensor = torch.tensor(data_values)
    children_mask_tensor = torch.tensor(children_masks, dtype=torch.uint8)
    weights_tensor = torch.tensor(weights_values)

    return make_octree(
        codes_tensor,
        data_tensor,
        children_mask_tensor,
        weights_tensor,
        maximum_depth=4,
    )
