#pragma once

#include <c10/macros/Macros.h>
#include <cmath>

#include "octree_build.h"
#include "octree_ray_marching.h"
#include "../coding/morton.h"

namespace torchscience::kernel::space_partitioning {

// Backward pass for octree_ray_marching (fixed-step mode)
//
// In fixed-step mode:
// - position[i] = origin + t[i] * direction
// - t[i] = t_start + i * step_size
//
// Gradients:
// - grad_origin: sum of grad_positions for valid samples
// - grad_direction: sum of t[i] * grad_positions for valid samples
// - grad_data: scatter-add of grad_output_data to sampled voxels
//
// In adaptive mode:
// - Positions depend discontinuously on ray parameters
// - Only grad_data is meaningful (grad_origin and grad_direction are zeroed)
//
// Parameters:
// - grad_positions: gradient of output positions, shape (maximum_steps, 3)
// - grad_output_data: gradient of output data, shape (maximum_steps, value_size)
// - mask: which samples were valid
// - data, codes, structure, children_mask: octree structure
// - capacity, value_size, maximum_depth: octree parameters
// - ox, oy, oz, dx, dy, dz: ray origin and direction (normalized)
// - t_start, t_end: marching bounds
// - use_fixed_step, step_size: stepping parameters
// - maximum_steps: number of output slots
// - out_grad_origin: output gradient for origin, shape (3,)
// - out_grad_direction: output gradient for direction, shape (3,)
// - found_indices: output array of voxel indices that were sampled (for grad_data scatter)
// - n_samples: output number of valid samples
//
template <typename scalar_t>
C10_HOST_DEVICE void octree_ray_march_backward(
    const scalar_t* grad_positions,
    const scalar_t* grad_output_data,
    const bool* mask,
    const scalar_t* data,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    int64_t maximum_depth,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float t_start, float t_end,
    bool use_fixed_step,
    float step_size,
    int64_t maximum_steps,
    scalar_t* out_grad_origin,
    scalar_t* out_grad_direction,
    int64_t* found_indices,
    int64_t* n_found
) {
    // Initialize outputs
    out_grad_origin[0] = scalar_t(0);
    out_grad_origin[1] = scalar_t(0);
    out_grad_origin[2] = scalar_t(0);
    out_grad_direction[0] = scalar_t(0);
    out_grad_direction[1] = scalar_t(0);
    out_grad_direction[2] = scalar_t(0);
    *n_found = 0;

    // Handle empty tree
    if (capacity == 0) {
        return;
    }

    // Look up root
    int64_t root_code = 0;
    int64_t root_idx = hash_lookup(structure, codes, capacity, root_code);
    if (root_idx < 0) {
        return;
    }

    if (use_fixed_step) {
        // Fixed-step mode: compute gradients for positions
        float t = t_start;
        int64_t sample_idx = 0;

        for (int64_t step = 0; step < maximum_steps && t < t_end; ++step) {
            if (!mask[step]) {
                t += step_size;
                continue;
            }

            float px = ox + t * dx;
            float py = oy + t * dy;
            float pz = oz + t * dz;

            // Find the voxel that was sampled (replay forward traversal)
            int64_t current_code = root_code;
            int64_t current_idx = root_idx;
            bool found = true;

            for (int64_t depth = 0; depth < maximum_depth && found; ++depth) {
                uint8_t child_mask = children_mask[current_idx];
                if (child_mask == 0) {
                    break;  // Leaf
                }

                int64_t octant = point_to_octant(px, py, pz, depth + 1);

                if (!(child_mask & (1 << octant))) {
                    found = false;
                    break;
                }

                int64_t child_code = coding::octree_child(current_code, octant);
                int64_t child_idx = hash_lookup(structure, codes, capacity, child_code);

                if (child_idx < 0) {
                    found = false;
                    break;
                }

                current_code = child_code;
                current_idx = child_idx;
            }

            if (found) {
                // Record voxel index for grad_data scatter
                found_indices[*n_found] = current_idx;
                (*n_found)++;

                // Gradient for origin: d(position)/d(origin) = I
                out_grad_origin[0] += grad_positions[step * 3 + 0];
                out_grad_origin[1] += grad_positions[step * 3 + 1];
                out_grad_origin[2] += grad_positions[step * 3 + 2];

                // Gradient for direction: d(position)/d(direction) = t * I
                out_grad_direction[0] += static_cast<scalar_t>(t) * grad_positions[step * 3 + 0];
                out_grad_direction[1] += static_cast<scalar_t>(t) * grad_positions[step * 3 + 1];
                out_grad_direction[2] += static_cast<scalar_t>(t) * grad_positions[step * 3 + 2];
            }

            t += step_size;
        }
    } else {
        // Adaptive mode: only find voxel indices for grad_data
        // Position gradients are zeroed (detached) due to discontinuous traversal
        TraversalStackEntry stack[MAX_STACK_DEPTH];
        int stack_size = 0;

        stack[stack_size++] = {root_code, t_start, t_end};
        int64_t sample_count = 0;

        while (stack_size > 0 && sample_count < maximum_steps) {
            TraversalStackEntry entry = stack[--stack_size];
            int64_t code = entry.code;
            float voxel_t_enter = entry.t_enter;
            float voxel_t_exit = entry.t_exit;

            if (voxel_t_exit < t_start || voxel_t_enter > t_end) {
                continue;
            }

            int64_t idx = hash_lookup(structure, codes, capacity, code);
            if (idx < 0) {
                continue;
            }

            uint8_t child_mask = children_mask[idx];
            int64_t depth = get_depth(code);

            if (child_mask == 0) {
                // Leaf node
                if (mask[sample_count]) {
                    found_indices[*n_found] = idx;
                    (*n_found)++;
                }
                sample_count++;
            } else {
                // Internal node - push children
                float voxel_size = 2.0f / static_cast<float>(1 << depth);
                float half_size = voxel_size * 0.5f;

                int64_t morton = code & 0x0FFFFFFFFFFFFFFFLL;
                int64_t ix, iy, iz;
                coding::morton_decode_3d(morton, ix, iy, iz);

                float cx = -1.0f + (static_cast<float>(ix) + 0.5f) * voxel_size;
                float cy = -1.0f + (static_cast<float>(iy) + 0.5f) * voxel_size;
                float cz = -1.0f + (static_cast<float>(iz) + 0.5f) * voxel_size;

                int ox_sign = dx >= 0 ? 0 : 1;
                int oy_sign = dy >= 0 ? 0 : 1;
                int oz_sign = dz >= 0 ? 0 : 1;

                for (int order = 7; order >= 0; --order) {
                    int octant = order ^ (ox_sign | (oy_sign << 1) | (oz_sign << 2));

                    if (!(child_mask & (1 << octant))) {
                        continue;
                    }

                    float child_x_min = (octant & 1) ? cx : cx - half_size;
                    float child_x_max = (octant & 1) ? cx + half_size : cx;
                    float child_y_min = (octant & 2) ? cy : cy - half_size;
                    float child_y_max = (octant & 2) ? cy + half_size : cy;
                    float child_z_min = (octant & 4) ? cz : cz - half_size;
                    float child_z_max = (octant & 4) ? cz + half_size : cz;

                    float child_t_enter, child_t_exit;
                    if (ray_voxel_intersect(
                            ox, oy, oz, dx, dy, dz,
                            child_x_min, child_x_max,
                            child_y_min, child_y_max,
                            child_z_min, child_z_max,
                            child_t_enter, child_t_exit
                        )) {
                        child_t_enter = fmaxf(child_t_enter, voxel_t_enter);
                        child_t_exit = fminf(child_t_exit, voxel_t_exit);

                        if (child_t_enter < child_t_exit && stack_size < MAX_STACK_DEPTH) {
                            int64_t child_code = coding::octree_child(code, octant);
                            stack[stack_size++] = {child_code, child_t_enter, child_t_exit};
                        }
                    }
                }
            }
        }
    }
}

}  // namespace torchscience::kernel::space_partitioning
