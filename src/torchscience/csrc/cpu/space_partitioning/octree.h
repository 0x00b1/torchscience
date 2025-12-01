#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../kernel/coding/morton.h"
#include "../../kernel/space_partitioning/octree_build.h"
#include "../../kernel/space_partitioning/octree_sample.h"
#include "../../kernel/space_partitioning/octree_sample_backward.h"
#include "../../kernel/space_partitioning/octree_ray_marching.h"
#include "../../kernel/space_partitioning/octree_ray_marching_backward.h"
#include "../../kernel/space_partitioning/octree_neighbors.h"

namespace torchscience::cpu::space_partitioning {

namespace {

using namespace torchscience::kernel::space_partitioning;
using namespace torchscience::kernel::coding;

// Build hash table with max displacement guarantee
// Returns true if successful, false if needs rebuild with larger capacity
template <typename T>
bool build_hash_table(
    const std::vector<int64_t>& codes,
    std::vector<int64_t>& structure,
    int64_t capacity,
    int64_t& max_displacement
) {
    std::fill(structure.begin(), structure.end(), -1);
    max_displacement = 0;

    for (size_t i = 0; i < codes.size(); ++i) {
        int64_t disp = hash_insert(structure.data(), codes.data(), capacity, codes[i], static_cast<int64_t>(i));
        if (disp < 0) {
            return false;  // Failed, need larger capacity
        }
        max_displacement = std::max(max_displacement, disp);
    }
    return max_displacement < MAX_PROBES;
}

}  // anonymous namespace

// Octree construction
// Returns: (codes, data, structure, children_mask, weights, maximum_depth, count)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
octree_build(
    const at::Tensor& points,
    const at::Tensor& data,
    int64_t maximum_depth,
    double capacity_factor,
    int64_t aggregation
) {
    TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
        "octree_build: points must be shape (n, 3), got ", points.sizes());
    TORCH_CHECK(data.dim() >= 1 && data.size(0) == points.size(0),
        "octree_build: data must have same first dimension as points");
    TORCH_CHECK(maximum_depth >= 1 && maximum_depth <= 15,
        "octree_build: maximum_depth must be in [1, 15], got ", maximum_depth);
    TORCH_CHECK(capacity_factor >= 1.0,
        "octree_build: capacity_factor must be >= 1.0, got ", capacity_factor);
    TORCH_CHECK(aggregation >= 0 && aggregation <= 2,
        "octree_build: aggregation must be 0 (mean), 1 (sum), or 2 (max)");

    int64_t n_points = points.size(0);
    auto agg_mode = static_cast<AggregationMode>(aggregation);

    // Get value shape (everything after first dimension)
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Make inputs contiguous
    at::Tensor points_cont = points.contiguous().to(at::kFloat);
    at::Tensor data_cont = data.contiguous();

    const float* points_ptr = points_cont.data_ptr<float>();

    // Handle empty input
    if (n_points == 0) {
        int64_t capacity = 1;
        return std::make_tuple(
            at::empty({0}, points.options().dtype(at::kLong)),
            at::empty({0, value_size}, data.options()),
            at::full({capacity}, -1, points.options().dtype(at::kLong)),
            at::empty({0}, points.options().dtype(at::kByte)),
            at::empty({0}, points.options().dtype(at::kFloat)),
            at::tensor(maximum_depth, points.options().dtype(at::kLong)),
            at::tensor(0, points.options().dtype(at::kLong))
        );
    }

    // Step 1: Compute leaf codes and group points by leaf
    // Use map for deterministic ordering
    std::map<int64_t, std::vector<int64_t>> leaf_points;  // code -> point indices

    for (int64_t i = 0; i < n_points; ++i) {
        float x = points_ptr[i * 3];
        float y = points_ptr[i * 3 + 1];
        float z = points_ptr[i * 3 + 2];
        int64_t code = point_to_code(x, y, z, maximum_depth);
        leaf_points[code].push_back(i);
    }

    // Step 2: Collect all unique ancestors (full hierarchy)
    std::set<int64_t> all_codes_set;

    for (const auto& [leaf_code, _] : leaf_points) {
        // Add leaf
        all_codes_set.insert(leaf_code);
        // Add all ancestors
        int64_t code = leaf_code;
        while (get_depth(code) > 0) {
            code = octree_parent(code);
            all_codes_set.insert(code);
        }
    }

    // Convert to sorted vector for deterministic indexing
    std::vector<int64_t> all_codes(all_codes_set.begin(), all_codes_set.end());
    int64_t n_nodes = static_cast<int64_t>(all_codes.size());

    // Create code -> index mapping
    std::map<int64_t, int64_t> code_to_idx;
    for (int64_t i = 0; i < n_nodes; ++i) {
        code_to_idx[all_codes[i]] = i;
    }

    // Step 3: Allocate output tensors
    at::Tensor out_codes = at::empty({n_nodes}, points.options().dtype(at::kLong));
    at::Tensor out_data = at::zeros({n_nodes, value_size}, data.options());
    at::Tensor out_children_mask = at::zeros({n_nodes}, points.options().dtype(at::kByte));
    at::Tensor out_weights = at::zeros({n_nodes}, points.options().dtype(at::kFloat));

    int64_t* codes_ptr = out_codes.data_ptr<int64_t>();
    uint8_t* mask_ptr = out_children_mask.data_ptr<uint8_t>();
    float* weights_ptr = out_weights.data_ptr<float>();

    // Copy codes
    for (int64_t i = 0; i < n_nodes; ++i) {
        codes_ptr[i] = all_codes[i];
    }

    // Step 4: Compute leaf data via aggregation and set children_mask
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_build_aggregate",
        [&] {
            const scalar_t* data_ptr = data_cont.data_ptr<scalar_t>();
            scalar_t* out_data_ptr = out_data.data_ptr<scalar_t>();

            // Process leaves: aggregate points
            for (const auto& [leaf_code, point_indices] : leaf_points) {
                int64_t idx = code_to_idx.at(leaf_code);
                int64_t n_pts = static_cast<int64_t>(point_indices.size());
                weights_ptr[idx] = static_cast<float>(n_pts);
                // children_mask stays 0 for leaves

                if (agg_mode == AggregationMode::MEAN || agg_mode == AggregationMode::SUM) {
                    // Sum all points
                    for (int64_t pt_idx : point_indices) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[idx * value_size + v] += data_ptr[pt_idx * value_size + v];
                        }
                    }
                    if (agg_mode == AggregationMode::MEAN && n_pts > 1) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[idx * value_size + v] /= static_cast<scalar_t>(n_pts);
                        }
                    }
                } else {  // MAX
                    // Initialize with first point
                    int64_t first_pt = point_indices[0];
                    for (int64_t v = 0; v < value_size; ++v) {
                        out_data_ptr[idx * value_size + v] = data_ptr[first_pt * value_size + v];
                    }
                    // Take max over remaining points
                    for (size_t p = 1; p < point_indices.size(); ++p) {
                        int64_t pt_idx = point_indices[p];
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[idx * value_size + v] = std::max(
                                out_data_ptr[idx * value_size + v],
                                data_ptr[pt_idx * value_size + v]
                            );
                        }
                    }
                }
            }

            // Build internal nodes: aggregate from children (bottom-up by depth)
            // Process depths from maximum_depth-1 down to 0
            for (int64_t depth = maximum_depth - 1; depth >= 0; --depth) {
                for (int64_t i = 0; i < n_nodes; ++i) {
                    int64_t code = all_codes[i];
                    if (get_depth(code) != depth) continue;

                    // This is an internal node - find children
                    float total_weight = 0.0f;
                    int64_t n_children = 0;
                    uint8_t child_mask = 0;

                    // Temporary for max aggregation
                    bool first_child = true;

                    for (int64_t octant = 0; octant < 8; ++octant) {
                        int64_t child_code = octree_child(code, octant);
                        auto it = code_to_idx.find(child_code);
                        if (it == code_to_idx.end()) continue;

                        int64_t child_idx = it->second;
                        child_mask |= (1 << octant);
                        float child_weight = weights_ptr[child_idx];
                        total_weight += child_weight;
                        n_children++;

                        if (agg_mode == AggregationMode::MEAN || agg_mode == AggregationMode::SUM) {
                            // Weighted sum
                            for (int64_t v = 0; v < value_size; ++v) {
                                if (agg_mode == AggregationMode::MEAN) {
                                    out_data_ptr[i * value_size + v] +=
                                        child_weight * out_data_ptr[child_idx * value_size + v];
                                } else {
                                    out_data_ptr[i * value_size + v] +=
                                        out_data_ptr[child_idx * value_size + v];
                                }
                            }
                        } else {  // MAX
                            if (first_child) {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] = out_data_ptr[child_idx * value_size + v];
                                }
                                first_child = false;
                            } else {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] = std::max(
                                        out_data_ptr[i * value_size + v],
                                        out_data_ptr[child_idx * value_size + v]
                                    );
                                }
                            }
                        }
                    }

                    mask_ptr[i] = child_mask;
                    weights_ptr[i] = total_weight;

                    // Normalize for mean
                    if (agg_mode == AggregationMode::MEAN && total_weight > 0) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[i * value_size + v] /= total_weight;
                        }
                    }
                }
            }
        }
    );

    // Step 5: Build hash table with max displacement guarantee
    int64_t capacity = next_power_of_2(static_cast<int64_t>(n_nodes * capacity_factor));
    if (capacity < 1) capacity = 1;

    std::vector<int64_t> structure(capacity, -1);
    int64_t max_displacement;

    while (!build_hash_table<int64_t>(all_codes, structure, capacity, max_displacement)) {
        capacity *= 2;
        structure.resize(capacity, -1);
    }

    at::Tensor out_structure = at::empty({capacity}, points.options().dtype(at::kLong));
    std::memcpy(out_structure.data_ptr<int64_t>(), structure.data(), capacity * sizeof(int64_t));

    // Reshape data if needed
    std::vector<int64_t> data_shape = {n_nodes};
    data_shape.insert(data_shape.end(), value_shape.begin(), value_shape.end());
    out_data = out_data.reshape(data_shape);

    return std::make_tuple(
        out_codes,
        out_data,
        out_structure,
        out_children_mask,
        out_weights,
        at::tensor(maximum_depth, points.options().dtype(at::kLong)),
        at::tensor(n_nodes, points.options().dtype(at::kLong))
    );
}

// Octree point sampling
// Returns: (data, found)
inline std::tuple<at::Tensor, at::Tensor>
octree_sample(
    const at::Tensor& data,
    const at::Tensor& codes,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& points,
    int64_t maximum_depth,
    int64_t interpolation,
    c10::optional<int64_t> query_depth_opt
) {
    int64_t query_depth = query_depth_opt.value_or(maximum_depth);

    TORCH_CHECK(points.dim() >= 1 && points.size(-1) == 3,
        "octree_sample: points last dimension must be 3, got ", points.size(-1));
    TORCH_CHECK(query_depth >= 0 && query_depth <= maximum_depth,
        "octree_sample: query_depth must be in [0, maximum_depth], got ", query_depth);
    TORCH_CHECK(interpolation == 0 || interpolation == 1,
        "octree_sample: interpolation must be 0 (nearest) or 1 (trilinear)");

    // Make inputs contiguous
    at::Tensor points_cont = points.contiguous().to(at::kFloat);
    at::Tensor data_cont = data.contiguous();
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor structure_cont = structure.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();

    // Compute output shapes
    std::vector<int64_t> points_shape;
    for (int64_t i = 0; i < points.dim() - 1; ++i) {
        points_shape.push_back(points.size(i));
    }
    int64_t n_queries = 1;
    for (auto s : points_shape) {
        n_queries *= s;
    }

    // Value shape from data
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Output shapes
    std::vector<int64_t> output_data_shape = points_shape;
    output_data_shape.insert(output_data_shape.end(), value_shape.begin(), value_shape.end());

    std::vector<int64_t> output_found_shape = points_shape;

    at::Tensor out_data = at::zeros(output_data_shape, data.options());
    at::Tensor out_found = at::zeros(output_found_shape, points.options().dtype(at::kBool));

    // Get pointers
    const float* points_ptr = points_cont.view({-1, 3}).data_ptr<float>();
    const int64_t* codes_ptr = codes_cont.data_ptr<int64_t>();
    const int64_t* structure_ptr = structure_cont.data_ptr<int64_t>();
    const uint8_t* mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    int64_t capacity = structure.size(0);

    bool* found_ptr = out_found.view({-1}).data_ptr<bool>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_sample_cpu",
        [&] {
            const scalar_t* data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* out_ptr = out_data.view({-1, value_size}).data_ptr<scalar_t>();

            at::parallel_for(0, n_queries, 1, [&](int64_t start, int64_t end) {
                // Thread-local buffer for trilinear
                std::vector<scalar_t> local_output(value_size);

                for (int64_t i = start; i < end; ++i) {
                    float x = points_ptr[i * 3];
                    float y = points_ptr[i * 3 + 1];
                    float z = points_ptr[i * 3 + 2];
                    bool found;

                    if (interpolation == 0) {
                        octree_sample_nearest(
                            data_ptr, codes_ptr, structure_ptr, mask_ptr,
                            capacity, value_size,
                            x, y, z, query_depth,
                            local_output.data(), &found
                        );
                    } else {
                        octree_sample_trilinear(
                            data_ptr, codes_ptr, structure_ptr, mask_ptr,
                            capacity, value_size,
                            x, y, z, query_depth,
                            local_output.data(), &found
                        );
                    }

                    found_ptr[i] = found;
                    for (int64_t v = 0; v < value_size; ++v) {
                        out_ptr[i * value_size + v] = local_output[v];
                    }
                }
            });
        }
    );

    return std::make_tuple(out_data, out_found);
}

// Octree point sampling backward pass
// Returns: (grad_data, grad_points)
inline std::tuple<at::Tensor, at::Tensor>
octree_sample_backward(
    const at::Tensor& grad_output,
    const at::Tensor& data,
    const at::Tensor& codes,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& points,
    int64_t maximum_depth,
    int64_t interpolation,
    c10::optional<int64_t> query_depth_opt
) {
    int64_t query_depth = query_depth_opt.value_or(maximum_depth);

    // Make inputs contiguous
    at::Tensor grad_output_cont = grad_output.contiguous();
    at::Tensor points_cont = points.contiguous().to(at::kFloat);
    at::Tensor data_cont = data.contiguous();
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor structure_cont = structure.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();

    // Compute shapes
    std::vector<int64_t> points_shape;
    for (int64_t i = 0; i < points.dim() - 1; ++i) {
        points_shape.push_back(points.size(i));
    }
    int64_t n_queries = 1;
    for (auto s : points_shape) {
        n_queries *= s;
    }

    // Value shape from data
    int64_t n_nodes = data.size(0);
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_size *= data.size(i);
    }

    // Allocate output gradients
    at::Tensor grad_data = at::zeros_like(data);
    at::Tensor grad_points = at::zeros_like(points);

    // Get pointers
    const float* points_ptr = points_cont.view({-1, 3}).data_ptr<float>();
    const int64_t* codes_ptr = codes_cont.data_ptr<int64_t>();
    const int64_t* structure_ptr = structure_cont.data_ptr<int64_t>();
    const uint8_t* mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    int64_t capacity = structure.size(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_sample_backward_cpu",
        [&] {
            const scalar_t* grad_output_ptr = grad_output_cont.view({-1, value_size}).data_ptr<scalar_t>();
            const scalar_t* data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* grad_data_ptr = grad_data.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* grad_points_ptr = grad_points.view({-1, 3}).data_ptr<scalar_t>();

            if (interpolation == 0) {
                // Nearest neighbor: only grad_data, no grad_points
                // Sequential to avoid race conditions on grad_data
                for (int64_t i = 0; i < n_queries; ++i) {
                    float x = points_ptr[i * 3];
                    float y = points_ptr[i * 3 + 1];
                    float z = points_ptr[i * 3 + 2];

                    int64_t found_idx;
                    octree_sample_nearest_backward(
                        grad_output_ptr + i * value_size,
                        codes_ptr, structure_ptr, mask_ptr,
                        capacity, value_size,
                        x, y, z, query_depth,
                        &found_idx
                    );

                    if (found_idx >= 0) {
                        // Scatter-add gradient to found voxel
                        for (int64_t v = 0; v < value_size; ++v) {
                            grad_data_ptr[found_idx * value_size + v] += grad_output_ptr[i * value_size + v];
                        }
                    }
                }
            } else {
                // Trilinear: grad_data and grad_points
                for (int64_t i = 0; i < n_queries; ++i) {
                    float x = points_ptr[i * 3];
                    float y = points_ptr[i * 3 + 1];
                    float z = points_ptr[i * 3 + 2];

                    int64_t corner_indices[8];
                    float corner_weights[8];
                    scalar_t grad_point[3];

                    octree_sample_trilinear_backward(
                        grad_output_ptr + i * value_size,
                        data_ptr,
                        codes_ptr, structure_ptr, mask_ptr,
                        capacity, value_size,
                        x, y, z, query_depth,
                        corner_indices,
                        corner_weights,
                        grad_point
                    );

                    // Scatter-add gradient to corner voxels
                    for (int c = 0; c < 8; ++c) {
                        if (corner_indices[c] >= 0) {
                            float w = corner_weights[c];
                            for (int64_t v = 0; v < value_size; ++v) {
                                grad_data_ptr[corner_indices[c] * value_size + v] +=
                                    static_cast<scalar_t>(w) * grad_output_ptr[i * value_size + v];
                            }
                        }
                    }

                    // Copy point gradient
                    grad_points_ptr[i * 3 + 0] = grad_point[0];
                    grad_points_ptr[i * 3 + 1] = grad_point[1];
                    grad_points_ptr[i * 3 + 2] = grad_point[2];
                }
            }
        }
    );

    return std::make_tuple(grad_data, grad_points);
}

// Octree ray marching
// Returns: (positions, data, mask)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
octree_ray_marching(
    const at::Tensor& data,
    const at::Tensor& codes,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& origins,
    const at::Tensor& directions,
    int64_t maximum_depth,
    c10::optional<double> step_size_opt,
    int64_t maximum_steps
) {
    TORCH_CHECK(origins.dim() == 2 && origins.size(1) == 3,
        "octree_ray_marching: origins must be shape (n_rays, 3), got ", origins.sizes());
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
        "octree_ray_marching: directions must be shape (n_rays, 3), got ", directions.sizes());
    TORCH_CHECK(origins.size(0) == directions.size(0),
        "octree_ray_marching: origins and directions must have same number of rays");
    TORCH_CHECK(maximum_steps > 0,
        "octree_ray_marching: maximum_steps must be positive, got ", maximum_steps);

    int64_t n_rays = origins.size(0);
    bool use_fixed_step = step_size_opt.has_value();
    float step_size = use_fixed_step ? static_cast<float>(step_size_opt.value()) : 0.0f;

    // Make inputs contiguous
    at::Tensor origins_cont = origins.contiguous().to(at::kFloat);
    at::Tensor directions_cont = directions.contiguous().to(at::kFloat);
    at::Tensor data_cont = data.contiguous();
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor structure_cont = structure.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();

    // Value shape from data
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Allocate outputs - use data dtype for positions and data to preserve precision
    at::Tensor out_positions = at::zeros({n_rays, maximum_steps, 3}, data.options());
    std::vector<int64_t> data_shape = {n_rays, maximum_steps};
    data_shape.insert(data_shape.end(), value_shape.begin(), value_shape.end());
    at::Tensor out_data = at::zeros(data_shape, data.options());
    at::Tensor out_mask = at::zeros({n_rays, maximum_steps}, data.options().dtype(at::kBool));

    // Get pointers
    const float* origins_ptr = origins_cont.data_ptr<float>();
    const float* directions_ptr = directions_cont.data_ptr<float>();
    const int64_t* codes_ptr = codes_cont.data_ptr<int64_t>();
    const int64_t* structure_ptr = structure_cont.data_ptr<int64_t>();
    const uint8_t* mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    int64_t capacity = structure.size(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_ray_marching_cpu",
        [&] {
            const scalar_t* data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* out_positions_ptr = out_positions.view({n_rays, maximum_steps * 3}).data_ptr<scalar_t>();
            scalar_t* out_data_ptr = out_data.view({n_rays, maximum_steps * value_size}).data_ptr<scalar_t>();
            bool* out_mask_ptr = out_mask.view({n_rays, maximum_steps}).data_ptr<bool>();

            at::parallel_for(0, n_rays, 1, [&](int64_t start, int64_t end) {
                // Thread-local buffers
                // Note: std::vector<bool> is a specialization without .data(), so use char
                std::vector<scalar_t> local_positions(maximum_steps * 3);
                std::vector<scalar_t> local_data(maximum_steps * value_size);
                std::vector<char> local_mask_storage(maximum_steps);

                for (int64_t ray_idx = start; ray_idx < end; ++ray_idx) {
                    float ox = origins_ptr[ray_idx * 3 + 0];
                    float oy = origins_ptr[ray_idx * 3 + 1];
                    float oz = origins_ptr[ray_idx * 3 + 2];

                    float dx = directions_ptr[ray_idx * 3 + 0];
                    float dy = directions_ptr[ray_idx * 3 + 1];
                    float dz = directions_ptr[ray_idx * 3 + 2];

                    // Normalize direction
                    float len = sqrtf(dx * dx + dy * dy + dz * dz);
                    if (len > 1e-8f) {
                        dx /= len;
                        dy /= len;
                        dz /= len;
                    }

                    // Ray-AABB intersection with [-1, 1]^3
                    float t_near, t_far;
                    bool hits_aabb = ray_aabb_intersect(ox, oy, oz, dx, dy, dz, t_near, t_far);

                    // Initialize local mask to false
                    std::fill(local_mask_storage.begin(), local_mask_storage.end(), 0);

                    // Cast to bool* for the kernel
                    bool* local_mask = reinterpret_cast<bool*>(local_mask_storage.data());

                    if (hits_aabb) {
                        // Start at entry point (or origin if inside)
                        float t_start = t_near > 0 ? t_near : 0.0f;

                        octree_ray_march(
                            data_ptr, codes_ptr, structure_ptr, mask_ptr,
                            capacity, value_size, maximum_depth,
                            ox, oy, oz, dx, dy, dz,
                            t_start, t_far,
                            use_fixed_step, step_size, maximum_steps,
                            local_positions.data(),
                            local_data.data(),
                            local_mask
                        );
                    }

                    // Copy to output
                    for (int64_t i = 0; i < maximum_steps; ++i) {
                        out_positions_ptr[ray_idx * maximum_steps * 3 + i * 3 + 0] = local_positions[i * 3 + 0];
                        out_positions_ptr[ray_idx * maximum_steps * 3 + i * 3 + 1] = local_positions[i * 3 + 1];
                        out_positions_ptr[ray_idx * maximum_steps * 3 + i * 3 + 2] = local_positions[i * 3 + 2];

                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[ray_idx * maximum_steps * value_size + i * value_size + v] =
                                local_data[i * value_size + v];
                        }

                        out_mask_ptr[ray_idx * maximum_steps + i] = local_mask[i];
                    }
                }
            });
        }
    );

    return std::make_tuple(out_positions, out_data, out_mask);
}

// Octree ray marching backward pass
// Returns: (grad_data, grad_origins, grad_directions)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor>
octree_ray_marching_backward(
    const at::Tensor& grad_positions,
    const at::Tensor& grad_data_out,
    const at::Tensor& mask,
    const at::Tensor& data,
    const at::Tensor& codes,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& origins,
    const at::Tensor& directions,
    int64_t maximum_depth,
    c10::optional<double> step_size_opt,
    int64_t maximum_steps
) {
    int64_t n_rays = origins.size(0);
    bool use_fixed_step = step_size_opt.has_value();
    float step_size = use_fixed_step ? static_cast<float>(step_size_opt.value()) : 0.0f;

    // Make inputs contiguous
    at::Tensor grad_positions_cont = grad_positions.contiguous();
    at::Tensor grad_data_out_cont = grad_data_out.contiguous();
    at::Tensor mask_cont = mask.contiguous();
    at::Tensor origins_cont = origins.contiguous().to(at::kFloat);
    at::Tensor directions_cont = directions.contiguous().to(at::kFloat);
    at::Tensor data_cont = data.contiguous();
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor structure_cont = structure.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();

    // Value shape from data
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_size *= data.size(i);
    }

    // Allocate output gradients
    at::Tensor grad_data = at::zeros_like(data);
    at::Tensor grad_origins = at::zeros_like(origins);
    at::Tensor grad_directions = at::zeros_like(directions);

    // Get pointers
    const float* origins_ptr = origins_cont.data_ptr<float>();
    const float* directions_ptr = directions_cont.data_ptr<float>();
    const int64_t* codes_ptr = codes_cont.data_ptr<int64_t>();
    const int64_t* structure_ptr = structure_cont.data_ptr<int64_t>();
    const uint8_t* children_mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    const bool* mask_ptr = mask_cont.data_ptr<bool>();
    int64_t capacity = structure.size(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_ray_marching_backward_cpu",
        [&] {
            const scalar_t* grad_positions_ptr = grad_positions_cont.view({n_rays, maximum_steps * 3}).data_ptr<scalar_t>();
            const scalar_t* grad_data_out_ptr = grad_data_out_cont.view({n_rays, maximum_steps * value_size}).data_ptr<scalar_t>();
            const scalar_t* data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* grad_data_ptr = grad_data.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* grad_origins_ptr = grad_origins.view({-1, 3}).data_ptr<scalar_t>();
            scalar_t* grad_directions_ptr = grad_directions.view({-1, 3}).data_ptr<scalar_t>();

            // Sequential to avoid race conditions on grad_data
            for (int64_t ray_idx = 0; ray_idx < n_rays; ++ray_idx) {
                float ox = origins_ptr[ray_idx * 3 + 0];
                float oy = origins_ptr[ray_idx * 3 + 1];
                float oz = origins_ptr[ray_idx * 3 + 2];

                float dx = directions_ptr[ray_idx * 3 + 0];
                float dy = directions_ptr[ray_idx * 3 + 1];
                float dz = directions_ptr[ray_idx * 3 + 2];

                // Normalize direction
                float len = sqrtf(dx * dx + dy * dy + dz * dz);
                if (len > 1e-8f) {
                    dx /= len;
                    dy /= len;
                    dz /= len;
                }

                // Ray-AABB intersection
                float t_near, t_far;
                bool hits_aabb = ray_aabb_intersect(ox, oy, oz, dx, dy, dz, t_near, t_far);

                if (!hits_aabb) {
                    continue;
                }

                float t_start = t_near > 0 ? t_near : 0.0f;

                // Get gradients and find voxel indices
                std::vector<int64_t> found_indices(maximum_steps);
                scalar_t local_grad_origin[3];
                scalar_t local_grad_direction[3];
                int64_t n_found = 0;

                octree_ray_march_backward(
                    grad_positions_ptr + ray_idx * maximum_steps * 3,
                    grad_data_out_ptr + ray_idx * maximum_steps * value_size,
                    mask_ptr + ray_idx * maximum_steps,
                    data_ptr,
                    codes_ptr, structure_ptr, children_mask_ptr,
                    capacity, value_size, maximum_depth,
                    ox, oy, oz, dx, dy, dz,
                    t_start, t_far,
                    use_fixed_step, step_size, maximum_steps,
                    local_grad_origin,
                    local_grad_direction,
                    found_indices.data(),
                    &n_found
                );

                // Store origin/direction gradients
                if (use_fixed_step) {
                    grad_origins_ptr[ray_idx * 3 + 0] = local_grad_origin[0];
                    grad_origins_ptr[ray_idx * 3 + 1] = local_grad_origin[1];
                    grad_origins_ptr[ray_idx * 3 + 2] = local_grad_origin[2];
                    grad_directions_ptr[ray_idx * 3 + 0] = local_grad_direction[0];
                    grad_directions_ptr[ray_idx * 3 + 1] = local_grad_direction[1];
                    grad_directions_ptr[ray_idx * 3 + 2] = local_grad_direction[2];
                }

                // Scatter-add grad_data_out to voxels
                int64_t sample_idx = 0;
                for (int64_t step = 0; step < maximum_steps; ++step) {
                    if (mask_ptr[ray_idx * maximum_steps + step] && sample_idx < n_found) {
                        int64_t voxel_idx = found_indices[sample_idx];
                        if (voxel_idx >= 0) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                grad_data_ptr[voxel_idx * value_size + v] +=
                                    grad_data_out_ptr[ray_idx * maximum_steps * value_size + step * value_size + v];
                            }
                        }
                        sample_idx++;
                    }
                }
            }
        }
    );

    return std::make_tuple(grad_data, grad_origins, grad_directions);
}

// Octree neighbor finding
// Returns: (neighbor_codes, neighbor_data)
inline std::tuple<at::Tensor, at::Tensor>
octree_neighbors(
    const at::Tensor& data,
    const at::Tensor& codes,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& query_codes,
    int64_t connectivity
) {
    TORCH_CHECK(connectivity == 6 || connectivity == 18 || connectivity == 26,
        "octree_neighbors: connectivity must be 6, 18, or 26, got ", connectivity);
    TORCH_CHECK(query_codes.dim() == 1,
        "octree_neighbors: query_codes must be 1D, got ", query_codes.dim(), " dims");

    int64_t n_queries = query_codes.size(0);

    // Make inputs contiguous
    at::Tensor data_cont = data.contiguous();
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor structure_cont = structure.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();
    at::Tensor query_codes_cont = query_codes.contiguous();

    // Value shape from data
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Allocate outputs
    at::Tensor out_codes = at::zeros({n_queries, connectivity}, query_codes.options());
    std::vector<int64_t> data_shape = {n_queries, connectivity};
    data_shape.insert(data_shape.end(), value_shape.begin(), value_shape.end());
    at::Tensor out_data = at::zeros(data_shape, data.options());

    // Get pointers
    const int64_t* codes_ptr = codes_cont.data_ptr<int64_t>();
    const int64_t* structure_ptr = structure_cont.data_ptr<int64_t>();
    const uint8_t* mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    const int64_t* query_codes_ptr = query_codes_cont.data_ptr<int64_t>();
    int64_t* out_codes_ptr = out_codes.data_ptr<int64_t>();
    int64_t capacity = structure.size(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_neighbors_cpu",
        [&] {
            const scalar_t* data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* out_data_ptr = out_data.view({n_queries, connectivity * value_size}).data_ptr<scalar_t>();

            at::parallel_for(0, n_queries, 1, [&](int64_t start, int64_t end) {
                // Thread-local buffers
                std::vector<int64_t> local_codes(connectivity);
                std::vector<scalar_t> local_data(connectivity * value_size);

                for (int64_t i = start; i < end; ++i) {
                    int64_t query_code = query_codes_ptr[i];

                    octree_find_neighbors(
                        data_ptr, codes_ptr, structure_ptr, mask_ptr,
                        capacity, value_size,
                        query_code, connectivity,
                        local_codes.data(),
                        local_data.data()
                    );

                    // Copy to output
                    for (int64_t n = 0; n < connectivity; ++n) {
                        out_codes_ptr[i * connectivity + n] = local_codes[n];
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[i * connectivity * value_size + n * value_size + v] =
                                local_data[n * value_size + v];
                        }
                    }
                }
            });
        }
    );

    // Reshape data back to include value dimensions
    std::vector<int64_t> final_data_shape = {n_queries, connectivity};
    final_data_shape.insert(final_data_shape.end(), value_shape.begin(), value_shape.end());
    out_data = out_data.view(final_data_shape);

    return std::make_tuple(out_codes, out_data);
}

// Octree insertion - insert new voxels into existing tree
// Returns: (codes, data, structure, children_mask, weights, maximum_depth, count)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
octree_insert(
    const at::Tensor& codes,
    const at::Tensor& data,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& weights,
    const at::Tensor& new_points,
    const at::Tensor& new_data,
    int64_t depth,
    int64_t maximum_depth,
    int64_t aggregation
) {
    TORCH_CHECK(new_points.dim() == 2 && new_points.size(1) == 3,
        "octree_insert: new_points must be shape (n, 3), got ", new_points.sizes());
    TORCH_CHECK(new_data.dim() >= 1 && new_data.size(0) == new_points.size(0),
        "octree_insert: new_data must have same first dimension as new_points");
    TORCH_CHECK(depth >= 1 && depth <= maximum_depth,
        "octree_insert: depth must be in [1, maximum_depth], got ", depth);
    TORCH_CHECK(aggregation >= 0 && aggregation <= 2,
        "octree_insert: aggregation must be 0 (mean), 1 (sum), or 2 (max)");

    auto agg_mode = static_cast<AggregationMode>(aggregation);

    // Make inputs contiguous
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor data_cont = data.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();
    at::Tensor weights_cont = weights.contiguous();
    at::Tensor new_points_cont = new_points.contiguous().to(at::kFloat);
    at::Tensor new_data_cont = new_data.contiguous();

    int64_t n_existing = codes.size(0);
    int64_t n_new = new_points.size(0);

    // Get value shape
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Handle empty insert
    if (n_new == 0) {
        return std::make_tuple(
            codes.clone(),
            data.clone(),
            structure.clone(),
            children_mask.clone(),
            weights.clone(),
            at::tensor(maximum_depth, codes.options()),
            at::tensor(n_existing, codes.options())
        );
    }

    const float* new_points_ptr = new_points_cont.data_ptr<float>();
    const int64_t* old_codes_ptr = codes_cont.data_ptr<int64_t>();
    const uint8_t* old_mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    const float* old_weights_ptr = weights_cont.data_ptr<float>();

    // Build existing code to index mapping
    std::map<int64_t, int64_t> existing_codes;
    for (int64_t i = 0; i < n_existing; ++i) {
        existing_codes[old_codes_ptr[i]] = i;
    }

    // Compute new leaf codes
    std::map<int64_t, std::vector<int64_t>> new_leaf_points;  // code -> point indices
    for (int64_t i = 0; i < n_new; ++i) {
        float x = new_points_ptr[i * 3];
        float y = new_points_ptr[i * 3 + 1];
        float z = new_points_ptr[i * 3 + 2];
        int64_t code = point_to_code(x, y, z, depth);
        new_leaf_points[code].push_back(i);
    }

    // Check for conflicts - can't insert into region with existing children
    for (const auto& [leaf_code, _] : new_leaf_points) {
        auto it = existing_codes.find(leaf_code);
        if (it != existing_codes.end()) {
            // Code exists - check if it's a leaf (OK to aggregate) or internal (conflict)
            int64_t existing_idx = it->second;
            if (old_mask_ptr[existing_idx] != 0) {
                TORCH_CHECK(false,
                    "octree_insert: cannot insert at depth ", depth,
                    " - existing internal node at this location. "
                    "Use octree_merge first to coarsen existing children.");
            }
        }
        // Also check if any ancestor of new leaf has the leaf as descendant
        // (would mean inserting into non-leaf region)
        int64_t check_code = leaf_code;
        while (get_depth(check_code) > 0) {
            check_code = octree_parent(check_code);
            auto parent_it = existing_codes.find(check_code);
            if (parent_it != existing_codes.end()) {
                int64_t parent_idx = parent_it->second;
                // Check if this ancestor has a child in the direction of our leaf
                int64_t child_octant = octree_octant(leaf_code);
                // Walk up to find the immediate child octant
                int64_t temp = leaf_code;
                while (get_depth(temp) > get_depth(check_code) + 1) {
                    temp = octree_parent(temp);
                }
                int64_t octant = octree_octant(temp);
                if (old_mask_ptr[parent_idx] & (1 << octant)) {
                    // Parent has a child in this octant but it's not at our depth
                    // This means we're trying to insert inside existing fine structure
                    // Actually this is OK if the child IS the leaf we're aggregating into
                    // Need to check if leaf_code itself exists
                    if (existing_codes.find(leaf_code) == existing_codes.end()) {
                        TORCH_CHECK(false,
                            "octree_insert: cannot insert at depth ", depth,
                            " - region already has finer structure. "
                            "Use octree_subdivide to create finer voxels first.");
                    }
                }
            }
        }
    }

    // Collect all codes for new tree: existing + new leaves + new ancestors
    std::set<int64_t> all_codes_set;

    // Add existing codes
    for (int64_t i = 0; i < n_existing; ++i) {
        all_codes_set.insert(old_codes_ptr[i]);
    }

    // Add new leaves and their ancestors
    for (const auto& [leaf_code, _] : new_leaf_points) {
        all_codes_set.insert(leaf_code);
        int64_t code = leaf_code;
        while (get_depth(code) > 0) {
            code = octree_parent(code);
            all_codes_set.insert(code);
        }
    }

    std::vector<int64_t> all_codes(all_codes_set.begin(), all_codes_set.end());
    int64_t n_nodes = static_cast<int64_t>(all_codes.size());

    // Create new code -> index mapping
    std::map<int64_t, int64_t> code_to_idx;
    for (int64_t i = 0; i < n_nodes; ++i) {
        code_to_idx[all_codes[i]] = i;
    }

    // Allocate outputs
    at::Tensor out_codes = at::empty({n_nodes}, codes.options());
    at::Tensor out_data = at::zeros({n_nodes, value_size}, data.options());
    at::Tensor out_children_mask = at::zeros({n_nodes}, codes.options().dtype(at::kByte));
    at::Tensor out_weights = at::zeros({n_nodes}, codes.options().dtype(at::kFloat));

    int64_t* out_codes_ptr = out_codes.data_ptr<int64_t>();
    uint8_t* out_mask_ptr = out_children_mask.data_ptr<uint8_t>();
    float* out_weights_ptr = out_weights.data_ptr<float>();

    // Copy codes
    for (int64_t i = 0; i < n_nodes; ++i) {
        out_codes_ptr[i] = all_codes[i];
    }

    // Copy existing data and weights for codes that existed before
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_insert_copy",
        [&] {
            const scalar_t* old_data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            const scalar_t* new_data_ptr = new_data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* out_data_ptr = out_data.data_ptr<scalar_t>();

            // Initialize from existing data
            for (const auto& [code, old_idx] : existing_codes) {
                int64_t new_idx = code_to_idx.at(code);
                for (int64_t v = 0; v < value_size; ++v) {
                    out_data_ptr[new_idx * value_size + v] = old_data_ptr[old_idx * value_size + v];
                }
                out_weights_ptr[new_idx] = old_weights_ptr[old_idx];
                out_mask_ptr[new_idx] = old_mask_ptr[old_idx];
            }

            // Process new leaf insertions - aggregate with existing if present
            for (const auto& [leaf_code, point_indices] : new_leaf_points) {
                int64_t new_idx = code_to_idx.at(leaf_code);
                auto existing_it = existing_codes.find(leaf_code);

                if (existing_it != existing_codes.end()) {
                    // Aggregate with existing leaf
                    float existing_weight = out_weights_ptr[new_idx];
                    float new_weight = static_cast<float>(point_indices.size());

                    // Compute new data from new points
                    scalar_t new_sum[64];  // Assume value_size <= 64
                    for (int64_t v = 0; v < value_size; ++v) {
                        new_sum[v] = scalar_t(0);
                    }
                    for (int64_t pt_idx : point_indices) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            new_sum[v] += new_data_ptr[pt_idx * value_size + v];
                        }
                    }

                    if (agg_mode == AggregationMode::MEAN) {
                        // Weighted mean
                        float total_weight = existing_weight + new_weight;
                        for (int64_t v = 0; v < value_size; ++v) {
                            scalar_t existing_val = out_data_ptr[new_idx * value_size + v];
                            scalar_t new_mean = new_sum[v] / static_cast<scalar_t>(new_weight);
                            out_data_ptr[new_idx * value_size + v] =
                                (existing_val * existing_weight + new_mean * new_weight) / total_weight;
                        }
                        out_weights_ptr[new_idx] = total_weight;
                    } else if (agg_mode == AggregationMode::SUM) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[new_idx * value_size + v] += new_sum[v];
                        }
                        out_weights_ptr[new_idx] = existing_weight + new_weight;
                    } else {  // MAX
                        for (int64_t pt_idx : point_indices) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[new_idx * value_size + v] = std::max(
                                    out_data_ptr[new_idx * value_size + v],
                                    new_data_ptr[pt_idx * value_size + v]
                                );
                            }
                        }
                        out_weights_ptr[new_idx] = existing_weight + new_weight;
                    }
                } else {
                    // New leaf - compute data from points
                    int64_t n_pts = static_cast<int64_t>(point_indices.size());
                    out_weights_ptr[new_idx] = static_cast<float>(n_pts);
                    out_mask_ptr[new_idx] = 0;  // leaf

                    if (agg_mode == AggregationMode::MEAN || agg_mode == AggregationMode::SUM) {
                        for (int64_t pt_idx : point_indices) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[new_idx * value_size + v] +=
                                    new_data_ptr[pt_idx * value_size + v];
                            }
                        }
                        if (agg_mode == AggregationMode::MEAN && n_pts > 1) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[new_idx * value_size + v] /= static_cast<scalar_t>(n_pts);
                            }
                        }
                    } else {  // MAX
                        int64_t first_pt = point_indices[0];
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[new_idx * value_size + v] =
                                new_data_ptr[first_pt * value_size + v];
                        }
                        for (size_t p = 1; p < point_indices.size(); ++p) {
                            int64_t pt_idx = point_indices[p];
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[new_idx * value_size + v] = std::max(
                                    out_data_ptr[new_idx * value_size + v],
                                    new_data_ptr[pt_idx * value_size + v]
                                );
                            }
                        }
                    }
                }
            }

            // Rebuild internal node data (bottom-up by depth)
            for (int64_t d = maximum_depth - 1; d >= 0; --d) {
                for (int64_t i = 0; i < n_nodes; ++i) {
                    int64_t code = all_codes[i];
                    if (get_depth(code) != d) continue;

                    // Check if this node or any of its descendants were affected
                    float total_weight = 0.0f;
                    uint8_t child_mask = 0;
                    bool first_child = true;

                    for (int64_t octant = 0; octant < 8; ++octant) {
                        int64_t child_code = octree_child(code, octant);
                        auto it = code_to_idx.find(child_code);
                        if (it == code_to_idx.end()) continue;

                        int64_t child_idx = it->second;
                        child_mask |= (1 << octant);
                        float child_weight = out_weights_ptr[child_idx];
                        total_weight += child_weight;

                        if (agg_mode == AggregationMode::MEAN) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[i * value_size + v] +=
                                    child_weight * out_data_ptr[child_idx * value_size + v];
                            }
                        } else if (agg_mode == AggregationMode::SUM) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[i * value_size + v] +=
                                    out_data_ptr[child_idx * value_size + v];
                            }
                        } else {  // MAX
                            if (first_child) {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] =
                                        out_data_ptr[child_idx * value_size + v];
                                }
                                first_child = false;
                            } else {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] = std::max(
                                        out_data_ptr[i * value_size + v],
                                        out_data_ptr[child_idx * value_size + v]
                                    );
                                }
                            }
                        }
                    }

                    out_mask_ptr[i] = child_mask;
                    out_weights_ptr[i] = total_weight;

                    if (agg_mode == AggregationMode::MEAN && total_weight > 0) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[i * value_size + v] /= total_weight;
                        }
                    }
                }
            }
        }
    );

    // Build hash table with max displacement guarantee
    int64_t capacity = next_power_of_2(static_cast<int64_t>(n_nodes * 2.0));
    if (capacity < 1) capacity = 1;

    std::vector<int64_t> new_structure(capacity, -1);
    int64_t max_displacement;

    while (!build_hash_table<int64_t>(all_codes, new_structure, capacity, max_displacement)) {
        capacity *= 2;
        new_structure.resize(capacity, -1);
    }

    at::Tensor out_structure = at::empty({capacity}, codes.options());
    std::memcpy(out_structure.data_ptr<int64_t>(), new_structure.data(), capacity * sizeof(int64_t));

    // Reshape data
    std::vector<int64_t> data_shape = {n_nodes};
    data_shape.insert(data_shape.end(), value_shape.begin(), value_shape.end());
    out_data = out_data.reshape(data_shape);

    return std::make_tuple(
        out_codes,
        out_data,
        out_structure,
        out_children_mask,
        out_weights,
        at::tensor(maximum_depth, codes.options()),
        at::tensor(n_nodes, codes.options())
    );
}

// Octree removal - remove voxels and prune empty ancestors
// Returns: (codes, data, structure, children_mask, weights, maximum_depth, count)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
octree_remove(
    const at::Tensor& codes,
    const at::Tensor& data,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& weights,
    const at::Tensor& remove_codes,
    int64_t maximum_depth,
    int64_t aggregation
) {
    TORCH_CHECK(remove_codes.dim() == 1,
        "octree_remove: remove_codes must be 1D, got ", remove_codes.dim(), "D");
    TORCH_CHECK(aggregation >= 0 && aggregation <= 2,
        "octree_remove: aggregation must be 0 (mean), 1 (sum), or 2 (max)");

    auto agg_mode = static_cast<AggregationMode>(aggregation);

    // Make inputs contiguous
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor data_cont = data.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();
    at::Tensor weights_cont = weights.contiguous();
    at::Tensor remove_codes_cont = remove_codes.contiguous();

    int64_t n_existing = codes.size(0);
    int64_t n_remove = remove_codes.size(0);

    // Get value shape
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Handle empty remove
    if (n_remove == 0) {
        return std::make_tuple(
            codes.clone(),
            data.clone(),
            structure.clone(),
            children_mask.clone(),
            weights.clone(),
            at::tensor(maximum_depth, codes.options()),
            at::tensor(n_existing, codes.options())
        );
    }

    const int64_t* old_codes_ptr = codes_cont.data_ptr<int64_t>();
    const uint8_t* old_mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    const float* old_weights_ptr = weights_cont.data_ptr<float>();
    const int64_t* remove_ptr = remove_codes_cont.data_ptr<int64_t>();

    // Build sets for lookup
    std::set<int64_t> codes_to_remove(remove_ptr, remove_ptr + n_remove);

    // Build existing code -> index mapping
    std::map<int64_t, int64_t> code_to_old_idx;
    for (int64_t i = 0; i < n_existing; ++i) {
        code_to_old_idx[old_codes_ptr[i]] = i;
    }

    // Verify all codes to remove exist and are leaves
    for (int64_t code : codes_to_remove) {
        auto it = code_to_old_idx.find(code);
        TORCH_CHECK(it != code_to_old_idx.end(),
            "octree_remove: code ", code, " not found in tree");
        TORCH_CHECK(old_mask_ptr[it->second] == 0,
            "octree_remove: can only remove leaf nodes, code ", code, " is internal");
    }

    // Collect remaining codes (after removal and pruning)
    std::set<int64_t> remaining_codes;
    for (int64_t i = 0; i < n_existing; ++i) {
        int64_t code = old_codes_ptr[i];
        if (codes_to_remove.find(code) == codes_to_remove.end()) {
            remaining_codes.insert(code);
        }
    }

    // Prune ancestors that become empty
    // An ancestor should be removed if all its children are removed
    bool changed = true;
    while (changed) {
        changed = false;
        std::set<int64_t> to_prune;
        for (int64_t code : remaining_codes) {
            if (get_depth(code) == 0) continue;  // Never prune root directly here

            // Check if this is an internal node with no remaining children
            bool is_internal = false;
            auto it = code_to_old_idx.find(code);
            if (it != code_to_old_idx.end()) {
                is_internal = (old_mask_ptr[it->second] != 0);
            }

            if (is_internal) {
                bool has_child = false;
                for (int64_t octant = 0; octant < 8; ++octant) {
                    int64_t child_code = octree_child(code, octant);
                    if (remaining_codes.find(child_code) != remaining_codes.end()) {
                        has_child = true;
                        break;
                    }
                }
                if (!has_child) {
                    to_prune.insert(code);
                    changed = true;
                }
            }
        }
        for (int64_t code : to_prune) {
            remaining_codes.erase(code);
        }
    }

    // Special case: if root has no children, remove entire tree
    int64_t root_code = octree_encode(0, 0, 0, 0);
    if (remaining_codes.find(root_code) != remaining_codes.end()) {
        bool root_has_child = false;
        for (int64_t octant = 0; octant < 8; ++octant) {
            int64_t child_code = octree_child(root_code, octant);
            if (remaining_codes.find(child_code) != remaining_codes.end()) {
                root_has_child = true;
                break;
            }
        }
        if (!root_has_child) {
            remaining_codes.erase(root_code);
        }
    }

    // Handle empty tree case
    if (remaining_codes.empty()) {
        int64_t capacity = 1;
        return std::make_tuple(
            at::empty({0}, codes.options()),
            at::empty({0, value_size}, data.options()),
            at::full({capacity}, -1, codes.options()),
            at::empty({0}, codes.options().dtype(at::kByte)),
            at::empty({0}, codes.options().dtype(at::kFloat)),
            at::tensor(maximum_depth, codes.options()),
            at::tensor(0, codes.options())
        );
    }

    // Build new tree
    std::vector<int64_t> all_codes(remaining_codes.begin(), remaining_codes.end());
    int64_t n_nodes = static_cast<int64_t>(all_codes.size());

    std::map<int64_t, int64_t> code_to_new_idx;
    for (int64_t i = 0; i < n_nodes; ++i) {
        code_to_new_idx[all_codes[i]] = i;
    }

    // Allocate outputs
    at::Tensor out_codes = at::empty({n_nodes}, codes.options());
    at::Tensor out_data = at::zeros({n_nodes, value_size}, data.options());
    at::Tensor out_children_mask = at::zeros({n_nodes}, codes.options().dtype(at::kByte));
    at::Tensor out_weights = at::zeros({n_nodes}, codes.options().dtype(at::kFloat));

    int64_t* out_codes_ptr = out_codes.data_ptr<int64_t>();
    uint8_t* out_mask_ptr = out_children_mask.data_ptr<uint8_t>();
    float* out_weights_ptr = out_weights.data_ptr<float>();

    for (int64_t i = 0; i < n_nodes; ++i) {
        out_codes_ptr[i] = all_codes[i];
    }

    // Copy data for leaves, recompute for internal nodes
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_remove_rebuild",
        [&] {
            const scalar_t* old_data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* out_data_ptr = out_data.data_ptr<scalar_t>();

            // First pass: copy leaf data
            for (int64_t i = 0; i < n_nodes; ++i) {
                int64_t code = all_codes[i];
                auto old_it = code_to_old_idx.find(code);
                if (old_it != code_to_old_idx.end()) {
                    int64_t old_idx = old_it->second;
                    if (old_mask_ptr[old_idx] == 0) {
                        // Leaf - copy data
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[i * value_size + v] = old_data_ptr[old_idx * value_size + v];
                        }
                        out_weights_ptr[i] = old_weights_ptr[old_idx];
                    }
                }
            }

            // Second pass: rebuild internal nodes bottom-up
            for (int64_t d = maximum_depth - 1; d >= 0; --d) {
                for (int64_t i = 0; i < n_nodes; ++i) {
                    int64_t code = all_codes[i];
                    if (get_depth(code) != d) continue;

                    float total_weight = 0.0f;
                    uint8_t child_mask = 0;
                    bool first_child = true;

                    for (int64_t octant = 0; octant < 8; ++octant) {
                        int64_t child_code = octree_child(code, octant);
                        auto it = code_to_new_idx.find(child_code);
                        if (it == code_to_new_idx.end()) continue;

                        int64_t child_idx = it->second;
                        child_mask |= (1 << octant);
                        float child_weight = out_weights_ptr[child_idx];
                        total_weight += child_weight;

                        if (agg_mode == AggregationMode::MEAN) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[i * value_size + v] +=
                                    child_weight * out_data_ptr[child_idx * value_size + v];
                            }
                        } else if (agg_mode == AggregationMode::SUM) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[i * value_size + v] +=
                                    out_data_ptr[child_idx * value_size + v];
                            }
                        } else {  // MAX
                            if (first_child) {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] =
                                        out_data_ptr[child_idx * value_size + v];
                                }
                                first_child = false;
                            } else {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] = std::max(
                                        out_data_ptr[i * value_size + v],
                                        out_data_ptr[child_idx * value_size + v]
                                    );
                                }
                            }
                        }
                    }

                    out_mask_ptr[i] = child_mask;
                    out_weights_ptr[i] = total_weight;

                    if (agg_mode == AggregationMode::MEAN && total_weight > 0) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[i * value_size + v] /= total_weight;
                        }
                    }
                }
            }
        }
    );

    // Build hash table
    int64_t capacity = next_power_of_2(static_cast<int64_t>(n_nodes * 2.0));
    if (capacity < 1) capacity = 1;

    std::vector<int64_t> new_structure(capacity, -1);
    int64_t max_displacement;

    while (!build_hash_table<int64_t>(all_codes, new_structure, capacity, max_displacement)) {
        capacity *= 2;
        new_structure.resize(capacity, -1);
    }

    at::Tensor out_structure = at::empty({capacity}, codes.options());
    std::memcpy(out_structure.data_ptr<int64_t>(), new_structure.data(), capacity * sizeof(int64_t));

    // Reshape data
    std::vector<int64_t> data_shape = {n_nodes};
    data_shape.insert(data_shape.end(), value_shape.begin(), value_shape.end());
    out_data = out_data.reshape(data_shape);

    return std::make_tuple(
        out_codes,
        out_data,
        out_structure,
        out_children_mask,
        out_weights,
        at::tensor(maximum_depth, codes.options()),
        at::tensor(n_nodes, codes.options())
    );
}

// Octree subdivision - split leaf voxel into 8 children
// Returns: (codes, data, structure, children_mask, weights, maximum_depth, count)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
octree_subdivide(
    const at::Tensor& codes,
    const at::Tensor& data,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& weights,
    const at::Tensor& subdivide_codes,
    int64_t maximum_depth,
    int64_t aggregation
) {
    TORCH_CHECK(subdivide_codes.dim() == 1,
        "octree_subdivide: subdivide_codes must be 1D");
    TORCH_CHECK(aggregation >= 0 && aggregation <= 2,
        "octree_subdivide: aggregation must be 0 (mean), 1 (sum), or 2 (max)");

    auto agg_mode = static_cast<AggregationMode>(aggregation);

    // Make inputs contiguous
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor data_cont = data.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();
    at::Tensor weights_cont = weights.contiguous();
    at::Tensor subdivide_codes_cont = subdivide_codes.contiguous();

    int64_t n_existing = codes.size(0);
    int64_t n_subdivide = subdivide_codes.size(0);

    // Get value shape
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Handle empty subdivide
    if (n_subdivide == 0) {
        return std::make_tuple(
            codes.clone(),
            data.clone(),
            structure.clone(),
            children_mask.clone(),
            weights.clone(),
            at::tensor(maximum_depth, codes.options()),
            at::tensor(n_existing, codes.options())
        );
    }

    const int64_t* old_codes_ptr = codes_cont.data_ptr<int64_t>();
    const uint8_t* old_mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    const float* old_weights_ptr = weights_cont.data_ptr<float>();
    const int64_t* subdivide_ptr = subdivide_codes_cont.data_ptr<int64_t>();

    // Build existing code -> index mapping
    std::map<int64_t, int64_t> code_to_old_idx;
    for (int64_t i = 0; i < n_existing; ++i) {
        code_to_old_idx[old_codes_ptr[i]] = i;
    }

    // Validate subdivide codes
    std::set<int64_t> codes_to_subdivide;
    for (int64_t i = 0; i < n_subdivide; ++i) {
        int64_t code = subdivide_ptr[i];
        auto it = code_to_old_idx.find(code);
        TORCH_CHECK(it != code_to_old_idx.end(),
            "octree_subdivide: code ", code, " not found in tree");
        TORCH_CHECK(old_mask_ptr[it->second] == 0,
            "octree_subdivide: can only subdivide leaf nodes, code ", code, " is internal");
        TORCH_CHECK(get_depth(code) < maximum_depth,
            "octree_subdivide: cannot subdivide at maximum_depth");
        codes_to_subdivide.insert(code);
    }

    // Collect all new codes
    std::set<int64_t> all_codes_set;
    for (int64_t i = 0; i < n_existing; ++i) {
        all_codes_set.insert(old_codes_ptr[i]);
    }

    // Add children for subdivided nodes
    for (int64_t code : codes_to_subdivide) {
        for (int64_t octant = 0; octant < 8; ++octant) {
            int64_t child_code = octree_child(code, octant);
            all_codes_set.insert(child_code);
        }
    }

    std::vector<int64_t> all_codes(all_codes_set.begin(), all_codes_set.end());
    int64_t n_nodes = static_cast<int64_t>(all_codes.size());

    std::map<int64_t, int64_t> code_to_new_idx;
    for (int64_t i = 0; i < n_nodes; ++i) {
        code_to_new_idx[all_codes[i]] = i;
    }

    // Allocate outputs
    at::Tensor out_codes = at::empty({n_nodes}, codes.options());
    at::Tensor out_data = at::zeros({n_nodes, value_size}, data.options());
    at::Tensor out_children_mask = at::zeros({n_nodes}, codes.options().dtype(at::kByte));
    at::Tensor out_weights = at::zeros({n_nodes}, codes.options().dtype(at::kFloat));

    int64_t* out_codes_ptr = out_codes.data_ptr<int64_t>();
    uint8_t* out_mask_ptr = out_children_mask.data_ptr<uint8_t>();
    float* out_weights_ptr = out_weights.data_ptr<float>();

    for (int64_t i = 0; i < n_nodes; ++i) {
        out_codes_ptr[i] = all_codes[i];
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_subdivide_rebuild",
        [&] {
            const scalar_t* old_data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* out_data_ptr = out_data.data_ptr<scalar_t>();

            // Copy existing node data
            for (int64_t i = 0; i < n_nodes; ++i) {
                int64_t code = all_codes[i];
                auto old_it = code_to_old_idx.find(code);
                if (old_it != code_to_old_idx.end()) {
                    int64_t old_idx = old_it->second;
                    for (int64_t v = 0; v < value_size; ++v) {
                        out_data_ptr[i * value_size + v] = old_data_ptr[old_idx * value_size + v];
                    }
                    out_weights_ptr[i] = old_weights_ptr[old_idx];
                    out_mask_ptr[i] = old_mask_ptr[old_idx];
                }
            }

            // Initialize children of subdivided nodes
            for (int64_t parent_code : codes_to_subdivide) {
                int64_t parent_new_idx = code_to_new_idx.at(parent_code);
                int64_t parent_old_idx = code_to_old_idx.at(parent_code);

                // Get parent data to inherit
                float parent_weight = old_weights_ptr[parent_old_idx];
                float child_weight = parent_weight / 8.0f;  // Split weight equally

                // Set parent as internal node
                out_mask_ptr[parent_new_idx] = 0xFF;  // All 8 children

                for (int64_t octant = 0; octant < 8; ++octant) {
                    int64_t child_code = octree_child(parent_code, octant);
                    int64_t child_idx = code_to_new_idx.at(child_code);

                    // Child inherits parent's data (for now - could be more sophisticated)
                    for (int64_t v = 0; v < value_size; ++v) {
                        out_data_ptr[child_idx * value_size + v] =
                            old_data_ptr[parent_old_idx * value_size + v];
                    }
                    out_weights_ptr[child_idx] = child_weight;
                    out_mask_ptr[child_idx] = 0;  // Children are leaves
                }
            }
        }
    );

    // Build hash table
    int64_t capacity = next_power_of_2(static_cast<int64_t>(n_nodes * 2.0));
    if (capacity < 1) capacity = 1;

    std::vector<int64_t> new_structure(capacity, -1);
    int64_t max_displacement;

    while (!build_hash_table<int64_t>(all_codes, new_structure, capacity, max_displacement)) {
        capacity *= 2;
        new_structure.resize(capacity, -1);
    }

    at::Tensor out_structure = at::empty({capacity}, codes.options());
    std::memcpy(out_structure.data_ptr<int64_t>(), new_structure.data(), capacity * sizeof(int64_t));

    // Reshape data
    std::vector<int64_t> data_shape = {n_nodes};
    data_shape.insert(data_shape.end(), value_shape.begin(), value_shape.end());
    out_data = out_data.reshape(data_shape);

    return std::make_tuple(
        out_codes,
        out_data,
        out_structure,
        out_children_mask,
        out_weights,
        at::tensor(maximum_depth, codes.options()),
        at::tensor(n_nodes, codes.options())
    );
}

// Octree merge - merge 8 sibling leaves into parent
// Returns: (codes, data, structure, children_mask, weights, maximum_depth, count)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
octree_merge(
    const at::Tensor& codes,
    const at::Tensor& data,
    const at::Tensor& structure,
    const at::Tensor& children_mask,
    const at::Tensor& weights,
    const at::Tensor& merge_codes,
    int64_t maximum_depth,
    int64_t aggregation
) {
    TORCH_CHECK(merge_codes.dim() == 1,
        "octree_merge: merge_codes must be 1D");
    TORCH_CHECK(aggregation >= 0 && aggregation <= 2,
        "octree_merge: aggregation must be 0 (mean), 1 (sum), or 2 (max)");

    auto agg_mode = static_cast<AggregationMode>(aggregation);

    // Make inputs contiguous
    at::Tensor codes_cont = codes.contiguous();
    at::Tensor data_cont = data.contiguous();
    at::Tensor children_mask_cont = children_mask.contiguous();
    at::Tensor weights_cont = weights.contiguous();
    at::Tensor merge_codes_cont = merge_codes.contiguous();

    int64_t n_existing = codes.size(0);
    int64_t n_merge = merge_codes.size(0);

    // Get value shape
    std::vector<int64_t> value_shape;
    int64_t value_size = 1;
    for (int64_t i = 1; i < data.dim(); ++i) {
        value_shape.push_back(data.size(i));
        value_size *= data.size(i);
    }

    // Handle empty merge
    if (n_merge == 0) {
        return std::make_tuple(
            codes.clone(),
            data.clone(),
            structure.clone(),
            children_mask.clone(),
            weights.clone(),
            at::tensor(maximum_depth, codes.options()),
            at::tensor(n_existing, codes.options())
        );
    }

    const int64_t* old_codes_ptr = codes_cont.data_ptr<int64_t>();
    const uint8_t* old_mask_ptr = children_mask_cont.data_ptr<uint8_t>();
    const float* old_weights_ptr = weights_cont.data_ptr<float>();
    const int64_t* merge_ptr = merge_codes_cont.data_ptr<int64_t>();

    // Build existing code -> index mapping
    std::map<int64_t, int64_t> code_to_old_idx;
    for (int64_t i = 0; i < n_existing; ++i) {
        code_to_old_idx[old_codes_ptr[i]] = i;
    }

    // For each merge code, find the parent and verify all 8 siblings exist as leaves
    std::set<int64_t> parents_to_convert;  // Parents that become leaves
    std::set<int64_t> children_to_remove;  // Children that get merged

    for (int64_t i = 0; i < n_merge; ++i) {
        int64_t code = merge_ptr[i];
        TORCH_CHECK(get_depth(code) > 0,
            "octree_merge: cannot merge at root level");

        int64_t parent_code = octree_parent(code);

        // Skip if already processed this parent
        if (parents_to_convert.find(parent_code) != parents_to_convert.end()) {
            continue;
        }

        // Verify parent exists
        auto parent_it = code_to_old_idx.find(parent_code);
        TORCH_CHECK(parent_it != code_to_old_idx.end(),
            "octree_merge: parent code not found");

        // Verify all 8 children exist and are leaves
        for (int64_t octant = 0; octant < 8; ++octant) {
            int64_t child_code = octree_child(parent_code, octant);
            auto child_it = code_to_old_idx.find(child_code);
            TORCH_CHECK(child_it != code_to_old_idx.end(),
                "octree_merge: all 8 siblings must exist, missing octant ", octant);
            TORCH_CHECK(old_mask_ptr[child_it->second] == 0,
                "octree_merge: all 8 siblings must be leaves, octant ", octant, " is internal");
            children_to_remove.insert(child_code);
        }

        parents_to_convert.insert(parent_code);
    }

    // Collect remaining codes (remove children, keep parents)
    std::set<int64_t> all_codes_set;
    for (int64_t i = 0; i < n_existing; ++i) {
        int64_t code = old_codes_ptr[i];
        if (children_to_remove.find(code) == children_to_remove.end()) {
            all_codes_set.insert(code);
        }
    }

    std::vector<int64_t> all_codes(all_codes_set.begin(), all_codes_set.end());
    int64_t n_nodes = static_cast<int64_t>(all_codes.size());

    std::map<int64_t, int64_t> code_to_new_idx;
    for (int64_t i = 0; i < n_nodes; ++i) {
        code_to_new_idx[all_codes[i]] = i;
    }

    // Allocate outputs
    at::Tensor out_codes = at::empty({n_nodes}, codes.options());
    at::Tensor out_data = at::zeros({n_nodes, value_size}, data.options());
    at::Tensor out_children_mask = at::zeros({n_nodes}, codes.options().dtype(at::kByte));
    at::Tensor out_weights = at::zeros({n_nodes}, codes.options().dtype(at::kFloat));

    int64_t* out_codes_ptr = out_codes.data_ptr<int64_t>();
    uint8_t* out_mask_ptr = out_children_mask.data_ptr<uint8_t>();
    float* out_weights_ptr = out_weights.data_ptr<float>();

    for (int64_t i = 0; i < n_nodes; ++i) {
        out_codes_ptr[i] = all_codes[i];
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        data_cont.scalar_type(),
        "octree_merge_rebuild",
        [&] {
            const scalar_t* old_data_ptr = data_cont.view({-1, value_size}).data_ptr<scalar_t>();
            scalar_t* out_data_ptr = out_data.data_ptr<scalar_t>();

            // Copy data for non-merged nodes
            for (int64_t i = 0; i < n_nodes; ++i) {
                int64_t code = all_codes[i];
                auto old_it = code_to_old_idx.find(code);
                if (old_it != code_to_old_idx.end()) {
                    int64_t old_idx = old_it->second;
                    // Skip parents that are being converted - will recompute
                    if (parents_to_convert.find(code) == parents_to_convert.end()) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[i * value_size + v] = old_data_ptr[old_idx * value_size + v];
                        }
                        out_weights_ptr[i] = old_weights_ptr[old_idx];
                        out_mask_ptr[i] = old_mask_ptr[old_idx];
                    }
                }
            }

            // Compute merged data for converted parents
            for (int64_t parent_code : parents_to_convert) {
                int64_t parent_new_idx = code_to_new_idx.at(parent_code);
                float total_weight = 0.0f;
                bool first_child = true;

                for (int64_t octant = 0; octant < 8; ++octant) {
                    int64_t child_code = octree_child(parent_code, octant);
                    int64_t child_old_idx = code_to_old_idx.at(child_code);
                    float child_weight = old_weights_ptr[child_old_idx];
                    total_weight += child_weight;

                    if (agg_mode == AggregationMode::MEAN) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[parent_new_idx * value_size + v] +=
                                child_weight * old_data_ptr[child_old_idx * value_size + v];
                        }
                    } else if (agg_mode == AggregationMode::SUM) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[parent_new_idx * value_size + v] +=
                                old_data_ptr[child_old_idx * value_size + v];
                        }
                    } else {  // MAX
                        if (first_child) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[parent_new_idx * value_size + v] =
                                    old_data_ptr[child_old_idx * value_size + v];
                            }
                            first_child = false;
                        } else {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[parent_new_idx * value_size + v] = std::max(
                                    out_data_ptr[parent_new_idx * value_size + v],
                                    old_data_ptr[child_old_idx * value_size + v]
                                );
                            }
                        }
                    }
                }

                out_weights_ptr[parent_new_idx] = total_weight;
                out_mask_ptr[parent_new_idx] = 0;  // Now a leaf

                if (agg_mode == AggregationMode::MEAN && total_weight > 0) {
                    for (int64_t v = 0; v < value_size; ++v) {
                        out_data_ptr[parent_new_idx * value_size + v] /= total_weight;
                    }
                }
            }

            // Rebuild ancestors of merged nodes
            for (int64_t d = maximum_depth - 1; d >= 0; --d) {
                for (int64_t i = 0; i < n_nodes; ++i) {
                    int64_t code = all_codes[i];
                    if (get_depth(code) != d) continue;
                    if (parents_to_convert.find(code) != parents_to_convert.end()) continue;

                    // Check if this is an internal node
                    auto old_it = code_to_old_idx.find(code);
                    if (old_it == code_to_old_idx.end()) continue;
                    if (old_mask_ptr[old_it->second] == 0) continue;  // Leaf

                    // Recompute from children
                    float total_weight = 0.0f;
                    uint8_t child_mask = 0;
                    bool first_child = true;

                    // Clear current data
                    for (int64_t v = 0; v < value_size; ++v) {
                        out_data_ptr[i * value_size + v] = scalar_t(0);
                    }

                    for (int64_t octant = 0; octant < 8; ++octant) {
                        int64_t child_code = octree_child(code, octant);
                        auto it = code_to_new_idx.find(child_code);
                        if (it == code_to_new_idx.end()) continue;

                        int64_t child_idx = it->second;
                        child_mask |= (1 << octant);
                        float child_weight = out_weights_ptr[child_idx];
                        total_weight += child_weight;

                        if (agg_mode == AggregationMode::MEAN) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[i * value_size + v] +=
                                    child_weight * out_data_ptr[child_idx * value_size + v];
                            }
                        } else if (agg_mode == AggregationMode::SUM) {
                            for (int64_t v = 0; v < value_size; ++v) {
                                out_data_ptr[i * value_size + v] +=
                                    out_data_ptr[child_idx * value_size + v];
                            }
                        } else {  // MAX
                            if (first_child) {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] =
                                        out_data_ptr[child_idx * value_size + v];
                                }
                                first_child = false;
                            } else {
                                for (int64_t v = 0; v < value_size; ++v) {
                                    out_data_ptr[i * value_size + v] = std::max(
                                        out_data_ptr[i * value_size + v],
                                        out_data_ptr[child_idx * value_size + v]
                                    );
                                }
                            }
                        }
                    }

                    out_mask_ptr[i] = child_mask;
                    out_weights_ptr[i] = total_weight;

                    if (agg_mode == AggregationMode::MEAN && total_weight > 0) {
                        for (int64_t v = 0; v < value_size; ++v) {
                            out_data_ptr[i * value_size + v] /= total_weight;
                        }
                    }
                }
            }
        }
    );

    // Build hash table
    int64_t capacity = next_power_of_2(static_cast<int64_t>(n_nodes * 2.0));
    if (capacity < 1) capacity = 1;

    std::vector<int64_t> new_structure(capacity, -1);
    int64_t max_displacement;

    while (!build_hash_table<int64_t>(all_codes, new_structure, capacity, max_displacement)) {
        capacity *= 2;
        new_structure.resize(capacity, -1);
    }

    at::Tensor out_structure = at::empty({capacity}, codes.options());
    std::memcpy(out_structure.data_ptr<int64_t>(), new_structure.data(), capacity * sizeof(int64_t));

    // Reshape data
    std::vector<int64_t> data_shape = {n_nodes};
    data_shape.insert(data_shape.end(), value_shape.begin(), value_shape.end());
    out_data = out_data.reshape(data_shape);

    return std::make_tuple(
        out_codes,
        out_data,
        out_structure,
        out_children_mask,
        out_weights,
        at::tensor(maximum_depth, codes.options()),
        at::tensor(n_nodes, codes.options())
    );
}

}  // namespace torchscience::cpu::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("octree_build", &torchscience::cpu::space_partitioning::octree_build);
    m.impl("octree_sample", &torchscience::cpu::space_partitioning::octree_sample);
    m.impl("octree_sample_backward", &torchscience::cpu::space_partitioning::octree_sample_backward);
    m.impl("octree_ray_marching", &torchscience::cpu::space_partitioning::octree_ray_marching);
    m.impl("octree_ray_marching_backward", &torchscience::cpu::space_partitioning::octree_ray_marching_backward);
    m.impl("octree_neighbors", &torchscience::cpu::space_partitioning::octree_neighbors);
    m.impl("octree_insert", &torchscience::cpu::space_partitioning::octree_insert);
    m.impl("octree_remove", &torchscience::cpu::space_partitioning::octree_remove);
    m.impl("octree_subdivide", &torchscience::cpu::space_partitioning::octree_subdivide);
    m.impl("octree_merge", &torchscience::cpu::space_partitioning::octree_merge);
}
