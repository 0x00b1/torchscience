#pragma once

#include <c10/macros/Macros.h>
#include <cmath>

#include "octree_build.h"
#include "../coding/morton.h"

namespace torchscience::kernel::space_partitioning {

// Maximum stack depth for hierarchical traversal (matches max tree depth of 15)
constexpr int64_t MAX_STACK_DEPTH = 16;

// Ray-AABB intersection for the [-1, 1]^3 bounding volume.
// Returns true if ray intersects, with t_near and t_far set to entry/exit distances.
// t_near < 0 means origin is inside the box.
C10_HOST_DEVICE inline bool ray_aabb_intersect(
    float ox, float oy, float oz,    // Ray origin
    float dx, float dy, float dz,    // Ray direction (normalized)
    float& t_near, float& t_far
) {
    constexpr float BOX_MIN = -1.0f;
    constexpr float BOX_MAX = 1.0f;
    constexpr float EPS = 1e-8f;

    t_near = -1e20f;
    t_far = 1e20f;

    // X axis
    if (fabsf(dx) < EPS) {
        // Ray parallel to X axis
        if (ox < BOX_MIN || ox > BOX_MAX) return false;
    } else {
        float inv_d = 1.0f / dx;
        float t0 = (BOX_MIN - ox) * inv_d;
        float t1 = (BOX_MAX - ox) * inv_d;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_near = t0 > t_near ? t0 : t_near;
        t_far = t1 < t_far ? t1 : t_far;
        if (t_near > t_far) return false;
    }

    // Y axis
    if (fabsf(dy) < EPS) {
        if (oy < BOX_MIN || oy > BOX_MAX) return false;
    } else {
        float inv_d = 1.0f / dy;
        float t0 = (BOX_MIN - oy) * inv_d;
        float t1 = (BOX_MAX - oy) * inv_d;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_near = t0 > t_near ? t0 : t_near;
        t_far = t1 < t_far ? t1 : t_far;
        if (t_near > t_far) return false;
    }

    // Z axis
    if (fabsf(dz) < EPS) {
        if (oz < BOX_MIN || oz > BOX_MAX) return false;
    } else {
        float inv_d = 1.0f / dz;
        float t0 = (BOX_MIN - oz) * inv_d;
        float t1 = (BOX_MAX - oz) * inv_d;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_near = t0 > t_near ? t0 : t_near;
        t_far = t1 < t_far ? t1 : t_far;
        if (t_near > t_far) return false;
    }

    return t_far >= 0;  // Ray must exit in front of origin
}

// Compute voxel bounds at given depth for a position
// Returns (voxel_min, voxel_max) in each dimension
C10_HOST_DEVICE inline void get_voxel_bounds(
    int64_t depth,
    float x, float y, float z,
    float& x_min, float& x_max,
    float& y_min, float& y_max,
    float& z_min, float& z_max
) {
    float scale = static_cast<float>(1 << depth);
    float voxel_size = 2.0f / scale;

    // Clamp to [-1, 1]
    x = fminf(fmaxf(x, -1.0f), 1.0f - 1e-6f);
    y = fminf(fmaxf(y, -1.0f), 1.0f - 1e-6f);
    z = fminf(fmaxf(z, -1.0f), 1.0f - 1e-6f);

    // Convert to voxel indices
    int64_t ix = static_cast<int64_t>((x + 1.0f) * 0.5f * scale);
    int64_t iy = static_cast<int64_t>((y + 1.0f) * 0.5f * scale);
    int64_t iz = static_cast<int64_t>((z + 1.0f) * 0.5f * scale);

    // Clamp indices
    ix = ix < 0 ? 0 : (ix >= (1 << depth) ? (1 << depth) - 1 : ix);
    iy = iy < 0 ? 0 : (iy >= (1 << depth) ? (1 << depth) - 1 : iy);
    iz = iz < 0 ? 0 : (iz >= (1 << depth) ? (1 << depth) - 1 : iz);

    x_min = -1.0f + static_cast<float>(ix) * voxel_size;
    x_max = x_min + voxel_size;
    y_min = -1.0f + static_cast<float>(iy) * voxel_size;
    y_max = y_min + voxel_size;
    z_min = -1.0f + static_cast<float>(iz) * voxel_size;
    z_max = z_min + voxel_size;
}

// Ray-voxel intersection: compute t values for ray entering/exiting a voxel
C10_HOST_DEVICE inline bool ray_voxel_intersect(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    float& t_enter, float& t_exit
) {
    constexpr float EPS = 1e-8f;

    t_enter = -1e20f;
    t_exit = 1e20f;

    // X axis
    if (fabsf(dx) < EPS) {
        if (ox < x_min || ox > x_max) return false;
    } else {
        float inv_d = 1.0f / dx;
        float t0 = (x_min - ox) * inv_d;
        float t1 = (x_max - ox) * inv_d;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = t0 > t_enter ? t0 : t_enter;
        t_exit = t1 < t_exit ? t1 : t_exit;
        if (t_enter > t_exit) return false;
    }

    // Y axis
    if (fabsf(dy) < EPS) {
        if (oy < y_min || oy > y_max) return false;
    } else {
        float inv_d = 1.0f / dy;
        float t0 = (y_min - oy) * inv_d;
        float t1 = (y_max - oy) * inv_d;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = t0 > t_enter ? t0 : t_enter;
        t_exit = t1 < t_exit ? t1 : t_exit;
        if (t_enter > t_exit) return false;
    }

    // Z axis
    if (fabsf(dz) < EPS) {
        if (oz < z_min || oz > z_max) return false;
    } else {
        float inv_d = 1.0f / dz;
        float t0 = (z_min - oz) * inv_d;
        float t1 = (z_max - oz) * inv_d;
        if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
        t_enter = t0 > t_enter ? t0 : t_enter;
        t_exit = t1 < t_exit ? t1 : t_exit;
        if (t_enter > t_exit) return false;
    }

    return true;
}

// Helper: compute octant for a point at a given depth
C10_HOST_DEVICE inline int64_t point_to_octant(float x, float y, float z, int64_t depth) {
    // Clamp to [-1, 1)
    x = fminf(fmaxf(x, -1.0f), 1.0f - 1e-6f);
    y = fminf(fmaxf(y, -1.0f), 1.0f - 1e-6f);
    z = fminf(fmaxf(z, -1.0f), 1.0f - 1e-6f);

    float scale = static_cast<float>(1 << depth);

    // Convert to integer indices at this depth
    int64_t ix = static_cast<int64_t>((x + 1.0f) * 0.5f * scale);
    int64_t iy = static_cast<int64_t>((y + 1.0f) * 0.5f * scale);
    int64_t iz = static_cast<int64_t>((z + 1.0f) * 0.5f * scale);

    // Octant is determined by least significant bit at this depth
    return (ix & 1) | ((iy & 1) << 1) | ((iz & 1) << 2);
}

// Stack entry for hierarchical traversal
struct TraversalStackEntry {
    int64_t code;      // Morton code of the voxel
    float t_enter;     // Entry t value for this voxel
    float t_exit;      // Exit t value for this voxel
};

// Hierarchical DDA ray marching through the octree.
// Uses iterative traversal with explicit stack for CUDA compatibility.
// Returns number of samples taken.
template <typename scalar_t>
C10_HOST_DEVICE int64_t octree_ray_march(
    const scalar_t* data,
    const int64_t* codes,
    const int64_t* structure,
    const uint8_t* children_mask,
    int64_t capacity,
    int64_t value_size,
    int64_t maximum_depth,
    float ox, float oy, float oz,      // Ray origin
    float dx, float dy, float dz,      // Ray direction (normalized)
    float t_start, float t_end,        // Marching bounds
    bool use_fixed_step,               // Fixed vs adaptive stepping
    float step_size,                   // Fixed step size (if use_fixed_step)
    int64_t maximum_steps,             // Maximum samples
    scalar_t* out_positions,           // Output: positions (maximum_steps, 3)
    scalar_t* out_data,                // Output: data (maximum_steps, value_size)
    bool* out_mask                     // Output: valid mask (maximum_steps,)
) {
    // Initialize output mask to false
    for (int64_t i = 0; i < maximum_steps; ++i) {
        out_mask[i] = false;
    }

    // Handle empty tree
    if (capacity == 0) {
        return 0;
    }

    // Look up root
    int64_t root_code = 0;  // depth=0, morton=0
    int64_t root_idx = hash_lookup(structure, codes, capacity, root_code);
    if (root_idx < 0) {
        return 0;  // Empty tree
    }

    int64_t n_samples = 0;

    if (use_fixed_step) {
        // Fixed-step mode: simple stepping along ray
        float t = t_start;

        for (int64_t step = 0; step < maximum_steps && t < t_end; ++step) {
            float px = ox + t * dx;
            float py = oy + t * dy;
            float pz = oz + t * dz;

            // Query octree at this position using top-down traversal
            int64_t current_code = root_code;
            int64_t current_idx = root_idx;
            bool found = true;

            for (int64_t depth = 0; depth < maximum_depth && found; ++depth) {
                uint8_t mask = children_mask[current_idx];
                if (mask == 0) {
                    // Leaf node - stop here
                    break;
                }

                // Compute which octant the point is in at depth+1
                int64_t octant = point_to_octant(px, py, pz, depth + 1);

                if (!(mask & (1 << octant))) {
                    // Child doesn't exist - empty region
                    found = false;
                    break;
                }

                // Look up child
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
                // Record sample
                out_positions[n_samples * 3 + 0] = static_cast<scalar_t>(px);
                out_positions[n_samples * 3 + 1] = static_cast<scalar_t>(py);
                out_positions[n_samples * 3 + 2] = static_cast<scalar_t>(pz);

                for (int64_t v = 0; v < value_size; ++v) {
                    out_data[n_samples * value_size + v] = data[current_idx * value_size + v];
                }

                out_mask[n_samples] = true;
                n_samples++;
            }

            t += step_size;
        }
    } else {
        // Adaptive stepping using hierarchical DDA
        // Use explicit stack for iterative traversal
        TraversalStackEntry stack[MAX_STACK_DEPTH];
        int stack_size = 0;

        // Push root
        stack[stack_size++] = {root_code, t_start, t_end};

        while (stack_size > 0 && n_samples < maximum_steps) {
            // Pop from stack
            TraversalStackEntry entry = stack[--stack_size];
            int64_t code = entry.code;
            float voxel_t_enter = entry.t_enter;
            float voxel_t_exit = entry.t_exit;

            // Skip if we've passed this voxel
            if (voxel_t_exit < t_start || voxel_t_enter > t_end) {
                continue;
            }

            // Look up voxel
            int64_t idx = hash_lookup(structure, codes, capacity, code);
            if (idx < 0) {
                continue;  // Voxel doesn't exist
            }

            uint8_t mask = children_mask[idx];
            int64_t depth = get_depth(code);

            if (mask == 0) {
                // Leaf node - sample at entry point
                float t = fmaxf(voxel_t_enter, t_start);
                if (t < t_end && n_samples < maximum_steps) {
                    float px = ox + t * dx;
                    float py = oy + t * dy;
                    float pz = oz + t * dz;

                    out_positions[n_samples * 3 + 0] = static_cast<scalar_t>(px);
                    out_positions[n_samples * 3 + 1] = static_cast<scalar_t>(py);
                    out_positions[n_samples * 3 + 2] = static_cast<scalar_t>(pz);

                    for (int64_t v = 0; v < value_size; ++v) {
                        out_data[n_samples * value_size + v] = data[idx * value_size + v];
                    }

                    out_mask[n_samples] = true;
                    n_samples++;
                }
            } else {
                // Internal node - push children that intersect the ray
                // Process children in back-to-front order based on ray direction
                // so front children are popped first (LIFO stack)

                float voxel_size = 2.0f / static_cast<float>(1 << depth);
                float half_size = voxel_size * 0.5f;

                // Get voxel center
                int64_t morton = code & 0x0FFFFFFFFFFFFFFFLL;
                int64_t ix, iy, iz;
                coding::morton_decode_3d(morton, ix, iy, iz);

                float cx = -1.0f + (static_cast<float>(ix) + 0.5f) * voxel_size;
                float cy = -1.0f + (static_cast<float>(iy) + 0.5f) * voxel_size;
                float cz = -1.0f + (static_cast<float>(iz) + 0.5f) * voxel_size;

                // Determine child order based on ray direction
                // We want to push far children first so near children are popped first
                int ox_sign = dx >= 0 ? 0 : 1;
                int oy_sign = dy >= 0 ? 0 : 1;
                int oz_sign = dz >= 0 ? 0 : 1;

                // Process all 8 children, back-to-front
                for (int order = 7; order >= 0; --order) {
                    // XOR with ray direction signs to get back-to-front order
                    int octant = order ^ (ox_sign | (oy_sign << 1) | (oz_sign << 2));

                    if (!(mask & (1 << octant))) {
                        continue;  // Child doesn't exist
                    }

                    // Compute child bounds
                    float child_x_min = (octant & 1) ? cx : cx - half_size;
                    float child_x_max = (octant & 1) ? cx + half_size : cx;
                    float child_y_min = (octant & 2) ? cy : cy - half_size;
                    float child_y_max = (octant & 2) ? cy + half_size : cy;
                    float child_z_min = (octant & 4) ? cz : cz - half_size;
                    float child_z_max = (octant & 4) ? cz + half_size : cz;

                    // Test ray-child intersection
                    float child_t_enter, child_t_exit;
                    if (ray_voxel_intersect(
                            ox, oy, oz, dx, dy, dz,
                            child_x_min, child_x_max,
                            child_y_min, child_y_max,
                            child_z_min, child_z_max,
                            child_t_enter, child_t_exit
                        )) {
                        // Clamp to parent bounds
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

    return n_samples;
}

}  // namespace torchscience::kernel::space_partitioning
