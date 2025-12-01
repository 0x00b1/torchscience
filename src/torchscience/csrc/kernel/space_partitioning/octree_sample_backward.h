#pragma once

#include <c10/macros/Macros.h>
#include <cstdint>
#include <cmath>

#include "../coding/morton.h"
#include "octree_build.h"
#include "octree_sample.h"

namespace torchscience::kernel::space_partitioning {

// Backward pass for octree_sample_nearest
//
// For nearest neighbor sampling, the forward pass returns data[idx] where idx
// is the voxel found during top-down traversal. The gradient is a scatter-add
// of grad_output to grad_data[idx].
//
// No gradient w.r.t. points (discrete lookup).
//
// Parameters:
// - grad_output: gradient of output data, shape (value_size,)
// - codes, structure, children_mask: octree structure (same as forward)
// - capacity, value_size: octree parameters
// - x, y, z: query point coordinates
// - query_depth: maximum depth to query
// - grad_data: output gradient array, shape (count, value_size) - ATOMIC ADD
// - found_idx: output - the voxel index that was found (-1 if not found)
//
template <typename scalar_t>
C10_HOST_DEVICE void octree_sample_nearest_backward(
    const scalar_t* grad_output,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    float x, float y, float z,
    int64_t query_depth,
    int64_t* found_idx
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
        *found_idx = -1;
        return;
    }

    // Top-down traversal from root to query_depth
    for (int64_t d = 0; d < query_depth; ++d) {
        uint8_t mask = children_mask[current_idx];

        // If current node is a leaf (no children), this is the target
        if (mask == 0) {
            *found_idx = current_idx;
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
            // Empty region - no gradient
            *found_idx = -1;
            return;
        }

        // Look up child node
        int64_t child_code = coding::octree_child(current_code, octant);
        int64_t child_idx = hash_lookup(structure, codes, capacity, child_code);

        if (child_idx < 0) {
            *found_idx = -1;
            return;
        }

        current_code = child_code;
        current_idx = child_idx;
    }

    // Reached query_depth - this is the target
    *found_idx = current_idx;
}

// Backward pass for octree_sample_trilinear
//
// For trilinear interpolation:
// output = sum_{c=0}^{7} weights[c] * data[corner_idx[c]]
//
// Gradient w.r.t. data[corner_idx[c]] = weights[c] * grad_output
// Gradient w.r.t. points involves d(weights)/d(x,y,z) * data[corner]
//
// Parameters:
// - grad_output: gradient of output data, shape (value_size,)
// - data: voxel data (for computing point gradients)
// - codes, structure, children_mask: octree structure
// - capacity, value_size: octree parameters
// - x, y, z: query point coordinates
// - query_depth: maximum depth to query
// - corner_indices: output array of 8 corner voxel indices (-1 if not found)
// - corner_weights: output array of 8 trilinear weights
// - grad_point: output gradient w.r.t. (x, y, z), shape (3,)
//
template <typename scalar_t>
C10_HOST_DEVICE void octree_sample_trilinear_backward(
    const scalar_t* grad_output,
    const scalar_t* data,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    float x, float y, float z,
    int64_t query_depth,
    int64_t* corner_indices,
    float* corner_weights,
    scalar_t* grad_point
) {
    // Clamp query point to [-1, 1]
    float x_clamped = x, y_clamped = y, z_clamped = z;
    if (x_clamped < -1.0f) x_clamped = -1.0f;
    if (x_clamped > 1.0f) x_clamped = 1.0f;
    if (y_clamped < -1.0f) y_clamped = -1.0f;
    if (y_clamped > 1.0f) y_clamped = 1.0f;
    if (z_clamped < -1.0f) z_clamped = -1.0f;
    if (z_clamped > 1.0f) z_clamped = 1.0f;

    // Compute cell-centered coordinates with half-cell offset
    float scale = static_cast<float>(1LL << query_depth);
    float normalized_x = (x_clamped + 1.0f) * 0.5f;  // [0, 1]
    float normalized_y = (y_clamped + 1.0f) * 0.5f;
    float normalized_z = (z_clamped + 1.0f) * 0.5f;

    // Cell-centered coordinate
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

    int64_t max_coord = (1LL << query_depth) - 1;

    // 8 corner offsets
    int64_t di[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    int64_t dj[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    int64_t dk[8] = {0, 0, 0, 0, 1, 1, 1, 1};

    // Compute weights and derivatives
    // Weight for corner (i, j, k) is:
    // w = (1-fx)^(1-i) * fx^i * (1-fy)^(1-j) * fy^j * (1-fz)^(1-k) * fz^k
    //
    // For gradient w.r.t. point:
    // d(fx)/d(x) = scale / 2 (from normalized_x = (x+1)/2, u = normalized_x * scale - 0.5)
    float dfdx = scale * 0.5f;  // d(fx)/d(x) = d(fy)/d(y) = d(fz)/d(z)

    // Initialize point gradient
    grad_point[0] = scalar_t(0);
    grad_point[1] = scalar_t(0);
    grad_point[2] = scalar_t(0);

    // Temporary buffer for corner data
    scalar_t corner_data[64];  // Assume value_size <= 64

    for (int64_t c = 0; c < 8; ++c) {
        int64_t ci = i0 + di[c];
        int64_t cj = j0 + dj[c];
        int64_t ck = k0 + dk[c];

        // Check bounds
        if (ci < 0 || ci > max_coord || cj < 0 || cj > max_coord || ck < 0 || ck > max_coord) {
            corner_indices[c] = -1;
            corner_weights[c] = 0.0f;
            continue;
        }

        // Convert cell index back to normalized coordinate
        float cx = -1.0f + (static_cast<float>(ci) + 0.5f) * (2.0f / scale);
        float cy = -1.0f + (static_cast<float>(cj) + 0.5f) * (2.0f / scale);
        float cz = -1.0f + (static_cast<float>(ck) + 0.5f) * (2.0f / scale);

        // Query this corner using nearest lookup
        int64_t idx;
        octree_sample_nearest_backward<scalar_t>(
            grad_output,  // unused in this call
            codes, structure, children_mask,
            capacity, value_size,
            cx, cy, cz, query_depth,
            &idx
        );

        corner_indices[c] = idx;

        // Compute weight for this corner
        float wx = (di[c] == 0) ? (1.0f - fx) : fx;
        float wy = (dj[c] == 0) ? (1.0f - fy) : fy;
        float wz = (dk[c] == 0) ? (1.0f - fz) : fz;
        corner_weights[c] = wx * wy * wz;

        // Compute gradient w.r.t. point for this corner
        if (idx >= 0) {
            // Get corner data for point gradient
            for (int64_t i = 0; i < value_size; ++i) {
                corner_data[i] = data[idx * value_size + i];
            }

            // d(weight)/d(fx) for this corner
            float dwx_dfx = (di[c] == 0) ? -1.0f : 1.0f;
            float dwy_dfy = (dj[c] == 0) ? -1.0f : 1.0f;
            float dwz_dfz = (dk[c] == 0) ? -1.0f : 1.0f;

            // d(weight)/d(x) = d(weight)/d(fx) * d(fx)/d(x)
            float dw_dx = dwx_dfx * wy * wz * dfdx;
            float dw_dy = wx * dwy_dfy * wz * dfdx;
            float dw_dz = wx * wy * dwz_dfz * dfdx;

            // grad_point += d(weight)/d(point) * dot(grad_output, corner_data)
            // But we want d(output)/d(point) where output = sum(weight * data)
            // d(output)/d(point) = sum(d(weight)/d(point) * data)
            // grad_loss_wrt_point = grad_output^T * d(output)/d(point)
            //                     = sum_v grad_output[v] * d(output[v])/d(point)
            //                     = sum_v grad_output[v] * sum_c d(weight[c])/d(point) * data[c,v]
            //                     = sum_c d(weight[c])/d(point) * sum_v grad_output[v] * data[c,v]
            scalar_t dot_product = scalar_t(0);
            for (int64_t i = 0; i < value_size; ++i) {
                dot_product += grad_output[i] * corner_data[i];
            }

            grad_point[0] += static_cast<scalar_t>(dw_dx) * dot_product;
            grad_point[1] += static_cast<scalar_t>(dw_dy) * dot_product;
            grad_point[2] += static_cast<scalar_t>(dw_dz) * dot_product;
        }
    }
}

}  // namespace torchscience::kernel::space_partitioning
