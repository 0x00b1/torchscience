#pragma once

#include <c10/macros/Macros.h>
#include <cmath>

#include "octree_build.h"
#include "octree_sample.h"
#include "../coding/morton.h"

namespace torchscience::kernel::space_partitioning {

// Neighbor offset directions for different connectivity levels
// Face neighbors (6): ±x, ±y, ±z
constexpr int FACE_NEIGHBOR_OFFSETS[6][3] = {
    {-1,  0,  0},  // -x
    { 1,  0,  0},  // +x
    { 0, -1,  0},  // -y
    { 0,  1,  0},  // +y
    { 0,  0, -1},  // -z
    { 0,  0,  1},  // +z
};

// Edge neighbors (12): combinations of two non-zero offsets
constexpr int EDGE_NEIGHBOR_OFFSETS[12][3] = {
    {-1, -1,  0},  // -x, -y
    {-1,  1,  0},  // -x, +y
    { 1, -1,  0},  // +x, -y
    { 1,  1,  0},  // +x, +y
    {-1,  0, -1},  // -x, -z
    {-1,  0,  1},  // -x, +z
    { 1,  0, -1},  // +x, -z
    { 1,  0,  1},  // +x, +z
    { 0, -1, -1},  // -y, -z
    { 0, -1,  1},  // -y, +z
    { 0,  1, -1},  // +y, -z
    { 0,  1,  1},  // +y, +z
};

// Corner neighbors (8): all three offsets non-zero
constexpr int CORNER_NEIGHBOR_OFFSETS[8][3] = {
    {-1, -1, -1},
    {-1, -1,  1},
    {-1,  1, -1},
    {-1,  1,  1},
    { 1, -1, -1},
    { 1, -1,  1},
    { 1,  1, -1},
    { 1,  1,  1},
};

// Get neighbor offset for a given connectivity and neighbor index
C10_HOST_DEVICE inline void get_neighbor_offset(
    int64_t connectivity,
    int64_t neighbor_idx,
    int& dx, int& dy, int& dz
) {
    if (neighbor_idx < 6) {
        // Face neighbor
        dx = FACE_NEIGHBOR_OFFSETS[neighbor_idx][0];
        dy = FACE_NEIGHBOR_OFFSETS[neighbor_idx][1];
        dz = FACE_NEIGHBOR_OFFSETS[neighbor_idx][2];
    } else if (neighbor_idx < 18) {
        // Edge neighbor
        int edge_idx = neighbor_idx - 6;
        dx = EDGE_NEIGHBOR_OFFSETS[edge_idx][0];
        dy = EDGE_NEIGHBOR_OFFSETS[edge_idx][1];
        dz = EDGE_NEIGHBOR_OFFSETS[edge_idx][2];
    } else {
        // Corner neighbor
        int corner_idx = neighbor_idx - 18;
        dx = CORNER_NEIGHBOR_OFFSETS[corner_idx][0];
        dy = CORNER_NEIGHBOR_OFFSETS[corner_idx][1];
        dz = CORNER_NEIGHBOR_OFFSETS[corner_idx][2];
    }
}

// Compute the center position of a voxel given its Morton code
C10_HOST_DEVICE inline void code_to_center(
    int64_t code,
    float& cx, float& cy, float& cz
) {
    int64_t depth = get_depth(code);
    int64_t morton = code & 0x0FFFFFFFFFFFFFFFLL;

    int64_t ix, iy, iz;
    coding::morton_decode_3d(morton, ix, iy, iz);

    float scale = static_cast<float>(1 << depth);
    float voxel_size = 2.0f / scale;

    cx = -1.0f + (static_cast<float>(ix) + 0.5f) * voxel_size;
    cy = -1.0f + (static_cast<float>(iy) + 0.5f) * voxel_size;
    cz = -1.0f + (static_cast<float>(iz) + 0.5f) * voxel_size;
}

// Find a single neighbor using top-down traversal
// Returns the neighbor code, or -1 if not found
template <typename scalar_t>
C10_HOST_DEVICE int64_t find_neighbor(
    const scalar_t* data,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    int64_t query_code,
    int dx, int dy, int dz,
    scalar_t* out_data
) {
    // Get query voxel info
    int64_t query_depth = get_depth(query_code);

    // Compute query voxel center
    float cx, cy, cz;
    code_to_center(query_code, cx, cy, cz);

    // Compute voxel size at query depth
    float scale = static_cast<float>(1 << query_depth);
    float voxel_size = 2.0f / scale;

    // Compute neighbor center position
    float nx = cx + static_cast<float>(dx) * voxel_size;
    float ny = cy + static_cast<float>(dy) * voxel_size;
    float nz = cz + static_cast<float>(dz) * voxel_size;

    // Check if neighbor is outside the domain
    if (nx < -1.0f || nx > 1.0f || ny < -1.0f || ny > 1.0f || nz < -1.0f || nz > 1.0f) {
        // Boundary - no neighbor exists
        for (int64_t v = 0; v < value_size; ++v) {
            out_data[v] = static_cast<scalar_t>(0);
        }
        return -1;
    }

    // Use top-down traversal to find the neighbor voxel
    // This is the same logic as octree_sample_nearest, limited to query_depth
    bool found;
    octree_sample_nearest(
        data, codes, structure, children_mask,
        capacity, value_size,
        nx, ny, nz, query_depth,
        out_data, &found
    );

    if (!found) {
        return -1;
    }

    // We need to return the code of the found voxel
    // Reconstruct the code by doing a separate lookup
    // (octree_sample_nearest doesn't return the code directly)

    // Look up root
    int64_t root_code = 0;
    int64_t root_idx = hash_lookup(structure, codes, capacity, root_code);
    if (root_idx < 0) {
        return -1;
    }

    // Traverse down to find the actual code
    int64_t current_code = root_code;
    int64_t current_idx = root_idx;

    for (int64_t d = 0; d < query_depth; ++d) {
        uint8_t mask = children_mask[current_idx];
        if (mask == 0) {
            // Leaf node - return this code
            return current_code;
        }

        // Compute octant for neighbor position at depth d+1
        float child_scale = static_cast<float>(1 << (d + 1));

        // Clamp to valid range
        float cnx = fminf(fmaxf(nx, -1.0f), 1.0f - 1e-6f);
        float cny = fminf(fmaxf(ny, -1.0f), 1.0f - 1e-6f);
        float cnz = fminf(fmaxf(nz, -1.0f), 1.0f - 1e-6f);

        int64_t ix = static_cast<int64_t>((cnx + 1.0f) * 0.5f * child_scale);
        int64_t iy = static_cast<int64_t>((cny + 1.0f) * 0.5f * child_scale);
        int64_t iz = static_cast<int64_t>((cnz + 1.0f) * 0.5f * child_scale);

        int64_t octant = (ix & 1) | ((iy & 1) << 1) | ((iz & 1) << 2);

        if (!(mask & (1 << octant))) {
            // Child doesn't exist - empty region
            return -1;
        }

        // Look up child
        int64_t child_code = coding::octree_child(current_code, octant);
        int64_t child_idx = hash_lookup(structure, codes, capacity, child_code);

        if (child_idx < 0) {
            return -1;
        }

        current_code = child_code;
        current_idx = child_idx;
    }

    return current_code;
}

// Find all neighbors for a single query voxel
template <typename scalar_t>
C10_HOST_DEVICE void octree_find_neighbors(
    const scalar_t* data,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    int64_t query_code,
    int64_t connectivity,
    int64_t* out_codes,      // Output: (connectivity,)
    scalar_t* out_data       // Output: (connectivity, value_size)
) {
    for (int64_t n = 0; n < connectivity; ++n) {
        int dx, dy, dz;
        get_neighbor_offset(connectivity, n, dx, dy, dz);

        out_codes[n] = find_neighbor(
            data, codes, structure, children_mask,
            capacity, value_size,
            query_code, dx, dy, dz,
            out_data + n * value_size
        );
    }
}

}  // namespace torchscience::kernel::space_partitioning
