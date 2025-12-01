#pragma once

#include <c10/macros/Macros.h>
#include <cstdint>

#include "../coding/morton.h"
#include "octree_build.h"

namespace torchscience::kernel::space_partitioning {

// Interpolation mode enum (must match Python constants)
enum class InterpolationMode : int64_t {
    NEAREST = 0,
    TRILINEAR = 1,
};

// Sample octree at point using top-down traversal with children_mask
//
// CRITICAL: This implements correct sparse semantics. We traverse from root
// to query_depth, checking children_mask at each level. If the query point's
// octant doesn't exist in children_mask, we return not-found (empty region).
// This prevents the ancestor-fallback bug where root would cover all queries.
//
// Traversal is bounded to query_depth levels (max 15), enabling torch.compile.
//
// Parameters:
// - data: voxel data array, shape (count, value_size)
// - codes: Morton codes array, shape (count,)
// - structure: hash table array, shape (capacity,)
// - children_mask: child existence bitmask, shape (count,), uint8
// - capacity: hash table capacity
// - value_size: number of elements per voxel
// - x, y, z: query point coordinates in [-1, 1]
// - query_depth: maximum depth to query
// - output: output data array (pre-allocated)
// - found: output boolean (did we find a voxel?)
//
template <typename scalar_t>
C10_HOST_DEVICE void octree_sample_nearest(
    const scalar_t* data,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    float x, float y, float z,
    int64_t query_depth,
    scalar_t* output,
    bool* found
) {
    // Clamp query point to [-1, 1]
    if (x < -1.0f) x = -1.0f;
    if (x > 1.0f) x = 1.0f;
    if (y < -1.0f) y = -1.0f;
    if (y > 1.0f) y = 1.0f;
    if (z < -1.0f) z = -1.0f;
    if (z > 1.0f) z = 1.0f;

    // Start at root (depth 0, morton 0)
    int64_t current_code = coding::octree_encode(0, 0, 0, 0);
    int64_t current_idx = hash_lookup(structure, codes, capacity, current_code);

    // Empty tree case
    if (current_idx < 0) {
        *found = false;
        for (int64_t i = 0; i < value_size; ++i) {
            output[i] = scalar_t(0);
        }
        return;
    }

    // Top-down traversal from root to query_depth
    for (int64_t d = 0; d < query_depth; ++d) {
        uint8_t mask = children_mask[current_idx];

        // If current node is a leaf (no children), return it
        // This handles merged coarse leaves correctly
        if (mask == 0) {
            *found = true;
            for (int64_t i = 0; i < value_size; ++i) {
                output[i] = data[current_idx * value_size + i];
            }
            return;
        }

        // Compute which octant the query point falls into at level d+1
        int64_t ix = quantize(x, d + 1);
        int64_t iy = quantize(y, d + 1);
        int64_t iz = quantize(z, d + 1);

        // Octant index from lowest bits of quantized coords
        int64_t octant = (ix & 1) | ((iy & 1) << 1) | ((iz & 1) << 2);

        // Check if this octant exists in children_mask
        if ((mask & (1 << octant)) == 0) {
            // Empty region - query point's path doesn't exist
            *found = false;
            for (int64_t i = 0; i < value_size; ++i) {
                output[i] = scalar_t(0);
            }
            return;
        }

        // Look up child node
        int64_t child_code = coding::octree_child(current_code, octant);
        int64_t child_idx = hash_lookup(structure, codes, capacity, child_code);

        // Consistency check (children_mask said it exists)
        if (child_idx < 0) {
            // Should not happen if tree is consistent
            *found = false;
            for (int64_t i = 0; i < value_size; ++i) {
                output[i] = scalar_t(0);
            }
            return;
        }

        current_code = child_code;
        current_idx = child_idx;
    }

    // Reached query_depth - return this node (leaf or internal)
    *found = true;
    for (int64_t i = 0; i < value_size; ++i) {
        output[i] = data[current_idx * value_size + i];
    }
}

// Sample octree with trilinear interpolation
// Uses cell-centered convention: values at voxel centers
//
// Each of the 8 interpolation corners is looked up using the same top-down
// traversal with children_mask (NOT ancestor fallback). This means:
// - Each corner query independently checks if its path exists via children_mask
// - If a corner's path doesn't exist, that corner contributes zero
// - found is True if ANY corner found a voxel; False if all corners miss
//
template <typename scalar_t>
C10_HOST_DEVICE void octree_sample_trilinear(
    const scalar_t* data,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    float x, float y, float z,
    int64_t query_depth,
    scalar_t* output,
    bool* found
) {
    // Clamp query point to [-1, 1]
    if (x < -1.0f) x = -1.0f;
    if (x > 1.0f) x = 1.0f;
    if (y < -1.0f) y = -1.0f;
    if (y > 1.0f) y = 1.0f;
    if (z < -1.0f) z = -1.0f;
    if (z > 1.0f) z = 1.0f;

    // Compute cell-centered coordinates with half-cell offset
    float scale = static_cast<float>(1LL << query_depth);
    float normalized_x = (x + 1.0f) * 0.5f;  // [0, 1]
    float normalized_y = (y + 1.0f) * 0.5f;
    float normalized_z = (z + 1.0f) * 0.5f;

    // Cell-centered coordinate (value locations are at center of each cell)
    float u = normalized_x * scale - 0.5f;
    float v = normalized_y * scale - 0.5f;
    float w = normalized_z * scale - 0.5f;

    // Integer indices of the 8 surrounding cell centers
    int64_t i0 = static_cast<int64_t>(floorf(u));
    int64_t j0 = static_cast<int64_t>(floorf(v));
    int64_t k0 = static_cast<int64_t>(floorf(w));

    // Fractional parts for interpolation weights
    float fx = u - static_cast<float>(i0);
    float fy = v - static_cast<float>(j0);
    float fz = w - static_cast<float>(k0);

    // Initialize output to zero
    for (int64_t i = 0; i < value_size; ++i) {
        output[i] = scalar_t(0);
    }

    *found = false;
    int64_t max_coord = (1LL << query_depth) - 1;

    // 8 corner weights (trilinear)
    float weights[8] = {
        (1.0f - fx) * (1.0f - fy) * (1.0f - fz),  // 000
        fx * (1.0f - fy) * (1.0f - fz),            // 100
        (1.0f - fx) * fy * (1.0f - fz),            // 010
        fx * fy * (1.0f - fz),                      // 110
        (1.0f - fx) * (1.0f - fy) * fz,            // 001
        fx * (1.0f - fy) * fz,                      // 101
        (1.0f - fx) * fy * fz,                      // 011
        fx * fy * fz,                               // 111
    };

    // 8 corner offsets
    int64_t di[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    int64_t dj[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    int64_t dk[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    // Temporary buffer for corner data
    scalar_t corner_data[64];  // Assume value_size <= 64

    for (int64_t c = 0; c < 8; ++c) {
        int64_t ci = i0 + di[c];
        int64_t cj = j0 + dj[c];
        int64_t ck = k0 + dk[c];

        // Check bounds
        if (ci < 0 || ci > max_coord || cj < 0 || cj > max_coord || ck < 0 || ck > max_coord) {
            // Out of bounds corner contributes zero
            continue;
        }

        // Convert cell index back to normalized coordinate
        float cx = -1.0f + (static_cast<float>(ci) + 0.5f) * (2.0f / scale);
        float cy = -1.0f + (static_cast<float>(cj) + 0.5f) * (2.0f / scale);
        float cz = -1.0f + (static_cast<float>(ck) + 0.5f) * (2.0f / scale);

        // Query this corner
        bool corner_found = false;
        octree_sample_nearest(
            data, codes, structure, children_mask,
            capacity, value_size,
            cx, cy, cz, query_depth,
            corner_data, &corner_found
        );

        if (corner_found) {
            *found = true;
            float w_c = weights[c];
            for (int64_t i = 0; i < value_size; ++i) {
                output[i] += static_cast<scalar_t>(w_c) * corner_data[i];
            }
        }
    }
}

}  // namespace torchscience::kernel::space_partitioning
