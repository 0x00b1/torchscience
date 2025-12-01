// src/torchscience/csrc/cpu/space_partitioning/kd_tree.h
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::space_partitioning {

namespace {

// Number of buckets for O(n) split evaluation
constexpr int64_t NUM_BUCKETS = 32;

// L1 Extent Heuristic for k-d tree splitting
// Uses bucket-based O(n) algorithm instead of O(n log n) sorting
// L1 extent (sum of bounding box edge lengths) is O(d) vs O(d²) for surface area
// and provides equivalent split quality for axis-aligned k-d tree queries
template <typename scalar_t>
struct L1ExtentSplitFinder {
    const scalar_t* points;
    int64_t n;
    int64_t d;
    const std::vector<int64_t>& indices;

    // Cost of traversing a node (inline function for Half/BFloat16 compatibility)
    static inline scalar_t traversal_cost() { return scalar_t(1); }
    // Cost of intersecting a point
    static inline scalar_t intersection_cost() { return scalar_t(1); }

    struct SplitResult {
        int64_t dim;
        scalar_t value;
        scalar_t cost;
    };

    struct Bucket {
        int64_t count = 0;
        std::vector<scalar_t> mins;
        std::vector<scalar_t> maxs;

        Bucket(int64_t d) : mins(d, std::numeric_limits<scalar_t>::max()),
                           maxs(d, std::numeric_limits<scalar_t>::lowest()) {}

        void add_point(const scalar_t* pt, int64_t d) {
            ++count;
            for (int64_t i = 0; i < d; ++i) {
                mins[i] = std::min(mins[i], pt[i]);
                maxs[i] = std::max(maxs[i], pt[i]);
            }
        }

        void merge(const Bucket& other) {
            count += other.count;
            for (size_t i = 0; i < mins.size(); ++i) {
                mins[i] = std::min(mins[i], other.mins[i]);
                maxs[i] = std::max(maxs[i], other.maxs[i]);
            }
        }
    };

    // Compute L1 extent (sum of bounding box edge lengths) - O(d) complexity
    // This is a proxy for query probability, equivalent to surface area for k-d trees
    static scalar_t compute_l1_extent(const std::vector<scalar_t>& mins,
                                      const std::vector<scalar_t>& maxs) {
        scalar_t extent = scalar_t(0);
        for (size_t i = 0; i < mins.size(); ++i) {
            extent += maxs[i] - mins[i];
        }
        return extent;
    }

    // Find optimal split using bucket-based heuristic with O(n) complexity
    SplitResult find_best_split() const {
        int64_t count = static_cast<int64_t>(indices.size());

        // Compute bounding box
        std::vector<scalar_t> mins(d, std::numeric_limits<scalar_t>::max());
        std::vector<scalar_t> maxs(d, std::numeric_limits<scalar_t>::lowest());

        for (int64_t idx : indices) {
            for (int64_t dim = 0; dim < d; ++dim) {
                scalar_t val = points[idx * d + dim];
                mins[dim] = std::min(mins[dim], val);
                maxs[dim] = std::max(maxs[dim], val);
            }
        }

        scalar_t parent_extent = compute_l1_extent(mins, maxs);
        if (parent_extent < scalar_t(1e-10)) parent_extent = scalar_t(1);  // Degenerate case

        SplitResult best{0, scalar_t(0), std::numeric_limits<scalar_t>::max()};

        // Try each dimension
        for (int64_t dim = 0; dim < d; ++dim) {
            scalar_t dim_min = mins[dim];
            scalar_t dim_max = maxs[dim];
            scalar_t dim_range = dim_max - dim_min;

            if (dim_range < scalar_t(1e-10)) continue;  // Skip degenerate dimension

            // Initialize buckets
            std::vector<Bucket> buckets(NUM_BUCKETS, Bucket(d));

            // O(n) bucket assignment
            for (int64_t idx : indices) {
                scalar_t val = points[idx * d + dim];
                int64_t bucket_idx = static_cast<int64_t>(
                    (val - dim_min) / dim_range * (NUM_BUCKETS - 1)
                );
                bucket_idx = std::clamp(bucket_idx, int64_t(0), NUM_BUCKETS - 1);
                buckets[bucket_idx].add_point(&points[idx * d], d);
            }

            // O(NUM_BUCKETS) prefix/suffix sweep
            // Build prefix sums (left side)
            std::vector<Bucket> prefix(NUM_BUCKETS, Bucket(d));
            prefix[0] = buckets[0];
            for (int64_t i = 1; i < NUM_BUCKETS; ++i) {
                prefix[i] = prefix[i - 1];
                prefix[i].merge(buckets[i]);
            }

            // Build suffix sums (right side) and evaluate cost
            Bucket suffix(d);
            for (int64_t i = NUM_BUCKETS - 1; i > 0; --i) {
                suffix.merge(buckets[i]);

                int64_t n_left = prefix[i - 1].count;
                int64_t n_right = suffix.count;

                if (n_left == 0 || n_right == 0) continue;

                scalar_t left_extent = compute_l1_extent(prefix[i - 1].mins, prefix[i - 1].maxs);
                scalar_t right_extent = compute_l1_extent(suffix.mins, suffix.maxs);

                // L1 extent cost (analogous to SAH)
                scalar_t cost = traversal_cost() +
                    (left_extent / parent_extent) * n_left * intersection_cost() +
                    (right_extent / parent_extent) * n_right * intersection_cost();

                if (cost < best.cost) {
                    best.dim = dim;
                    best.value = dim_min + (static_cast<scalar_t>(i) / NUM_BUCKETS) * dim_range;
                    best.cost = cost;
                }
            }
        }

        // Fallback: if no valid split found, use centroid of first dimension
        if (best.cost == std::numeric_limits<scalar_t>::max()) {
            best.dim = 0;
            best.value = (mins[0] + maxs[0]) / scalar_t(2);
            best.cost = count * intersection_cost();
        }

        return best;
    }
};

template <typename scalar_t>
struct KdTreeBuilder {
    std::vector<int64_t> split_dim;
    std::vector<scalar_t> split_val;  // Matches input dtype for precision
    std::vector<int64_t> left;
    std::vector<int64_t> right;
    std::vector<int64_t> leaf_starts;
    std::vector<int64_t> leaf_counts;
    std::vector<int64_t> final_indices;

    const at::Tensor& points;
    int64_t n;
    int64_t d;
    int64_t leaf_size;

    KdTreeBuilder(const at::Tensor& pts, int64_t ls)
        : points(pts), n(pts.size(0)), d(pts.size(1)), leaf_size(ls) {
        // Reserve estimated capacity to avoid reallocations during build
        // Upper bound: 2n-1 nodes for a complete binary tree
        int64_t est_nodes = 2 * n;
        int64_t est_leaves = (n + ls - 1) / ls;
        split_dim.reserve(est_nodes);
        split_val.reserve(est_nodes);
        left.reserve(est_nodes);
        right.reserve(est_nodes);
        leaf_starts.reserve(est_leaves);
        leaf_counts.reserve(est_leaves);
        final_indices.reserve(n);
    }

    int64_t build(std::vector<int64_t>& idx_subset, int64_t depth) {
        int64_t node_idx = static_cast<int64_t>(split_dim.size());
        int64_t count = static_cast<int64_t>(idx_subset.size());

        // Leaf node
        if (count <= leaf_size) {
            split_dim.push_back(-1);
            split_val.push_back(scalar_t(0));
            left.push_back(-1);
            right.push_back(-1);

            leaf_starts.push_back(static_cast<int64_t>(final_indices.size()));
            leaf_counts.push_back(count);
            for (int64_t idx : idx_subset) {
                final_indices.push_back(idx);
            }
            return node_idx;
        }

        // Use bucket-based L1 extent heuristic to find best split
        const scalar_t* pts_ptr = points.data_ptr<scalar_t>();
        L1ExtentSplitFinder<scalar_t> finder{pts_ptr, n, d, idx_subset};
        auto split = finder.find_best_split();

        int64_t split_d = split.dim;
        scalar_t split_v = split.value;

        // Partition points
        std::vector<int64_t> left_indices, right_indices;
        for (int64_t idx : idx_subset) {
            scalar_t val = pts_ptr[idx * d + split_d];
            if (val < split_v) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        // Handle edge case: all points on one side (use in-place median fallback)
        if (left_indices.empty() || right_indices.empty()) {
            // Use in-place nth_element with projection comparator - avoids allocation
            int64_t mid = count / 2;
            std::nth_element(
                idx_subset.begin(), idx_subset.begin() + mid, idx_subset.end(),
                [&](int64_t a, int64_t b) {
                    return pts_ptr[a * d + split_d] < pts_ptr[b * d + split_d];
                }
            );
            split_v = pts_ptr[idx_subset[mid] * d + split_d];

            // Partition in-place: left gets [0, mid), right gets [mid, count)
            left_indices.assign(idx_subset.begin(), idx_subset.begin() + mid);
            right_indices.assign(idx_subset.begin() + mid, idx_subset.end());
        }

        // Add node (placeholders for children)
        split_dim.push_back(split_d);
        split_val.push_back(split_v);
        left.push_back(-1);
        right.push_back(-1);

        // Recurse
        int64_t left_child = build(left_indices, depth + 1);
        int64_t right_child = build(right_indices, depth + 1);

        // Update children
        left[node_idx] = left_child;
        right[node_idx] = right_child;

        return node_idx;
    }
};

}  // anonymous namespace

// Single tree build (unbatched)
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
kd_tree_build(
    const at::Tensor& points,
    int64_t leaf_size
) {
    TORCH_CHECK(points.dim() == 2, "kd_tree_build: points must be 2D (n, d)");
    TORCH_CHECK(leaf_size > 0, "kd_tree_build: leaf_size must be > 0");

    int64_t n = points.size(0);
    int64_t d = points.size(1);

    if (n == 0) {
        return std::make_tuple(
            points.clone(),
            at::empty({0}, points.options().dtype(at::kLong)),
            at::empty({0}, points.options()),  // split_val matches input dtype
            at::empty({0}, points.options().dtype(at::kLong)),
            at::empty({0}, points.options().dtype(at::kLong)),
            at::empty({0}, points.options().dtype(at::kLong)),
            at::empty({0}, points.options().dtype(at::kLong)),
            at::empty({0}, points.options().dtype(at::kLong))
        );
    }

    at::Tensor points_contig = points.contiguous();

    std::vector<int64_t> initial_indices(n);
    for (int64_t i = 0; i < n; ++i) {
        initial_indices[i] = i;
    }

    at::Tensor split_dim_t, split_val_t, left_t, right_t, indices_t, leaf_starts_t, leaf_counts_t;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "kd_tree_build_cpu",
        [&]() {
            KdTreeBuilder<scalar_t> builder(points_contig, leaf_size);
            builder.build(initial_indices, 0);

            int64_t n_nodes = static_cast<int64_t>(builder.split_dim.size());
            int64_t n_leaves = static_cast<int64_t>(builder.leaf_starts.size());

            split_dim_t = at::empty({n_nodes}, points.options().dtype(at::kLong));
            split_val_t = at::empty({n_nodes}, points.options());  // Matches input dtype
            left_t = at::empty({n_nodes}, points.options().dtype(at::kLong));
            right_t = at::empty({n_nodes}, points.options().dtype(at::kLong));
            indices_t = at::empty({n}, points.options().dtype(at::kLong));
            leaf_starts_t = at::empty({n_leaves}, points.options().dtype(at::kLong));
            leaf_counts_t = at::empty({n_leaves}, points.options().dtype(at::kLong));

            std::memcpy(split_dim_t.data_ptr<int64_t>(), builder.split_dim.data(), n_nodes * sizeof(int64_t));
            std::memcpy(split_val_t.data_ptr<scalar_t>(), builder.split_val.data(), n_nodes * sizeof(scalar_t));
            std::memcpy(left_t.data_ptr<int64_t>(), builder.left.data(), n_nodes * sizeof(int64_t));
            std::memcpy(right_t.data_ptr<int64_t>(), builder.right.data(), n_nodes * sizeof(int64_t));
            std::memcpy(indices_t.data_ptr<int64_t>(), builder.final_indices.data(), n * sizeof(int64_t));
            std::memcpy(leaf_starts_t.data_ptr<int64_t>(), builder.leaf_starts.data(), n_leaves * sizeof(int64_t));
            std::memcpy(leaf_counts_t.data_ptr<int64_t>(), builder.leaf_counts.data(), n_leaves * sizeof(int64_t));
        }
    );

    return std::make_tuple(
        points_contig,
        split_dim_t,
        split_val_t,
        left_t,
        right_t,
        indices_t,
        leaf_starts_t,
        leaf_counts_t
    );
}

// Batched tree build using at::parallel_for with pre-padded output tensors
// Returns tuple of (B, max_*) tensors for efficient Python consumption
inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
kd_tree_build_batched(
    const at::Tensor& points,
    int64_t leaf_size
) {
    TORCH_CHECK(points.dim() == 3, "kd_tree_build_batched: points must be 3D (B, n, d)");
    TORCH_CHECK(leaf_size > 0, "kd_tree_build_batched: leaf_size must be > 0");

    int64_t B = points.size(0);
    int64_t n = points.size(1);
    int64_t d = points.size(2);

    if (B == 0 || n == 0) {
        return std::make_tuple(
            points.clone(),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options()),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, n}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options().dtype(at::kLong)),
            at::empty({B, 0}, points.options().dtype(at::kLong))
        );
    }

    at::Tensor points_contig = points.contiguous();

    // Phase 1: Build trees in parallel, store in thread-local vectors
    // Per-batch tree data with type-erased split_val storage
    struct TreeData {
        std::vector<int64_t> split_dim;
        std::vector<int64_t> left;
        std::vector<int64_t> right;
        std::vector<int64_t> indices;
        std::vector<int64_t> leaf_starts;
        std::vector<int64_t> leaf_counts;
    };
    std::vector<TreeData> tree_data(B);

    // Type-safe split_val storage - allocated inside dispatch
    // We use a shared_ptr to void with custom deleter for type erasure
    std::vector<std::shared_ptr<void>> split_val_storage(B);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "kd_tree_build_batched_cpu",
        [&]() {
            // Allocate typed vectors for each batch
            std::vector<std::vector<scalar_t>> typed_split_vals(B);

            at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    at::Tensor pts_b = points_contig[b];

                    std::vector<int64_t> initial_indices(n);
                    for (int64_t i = 0; i < n; ++i) initial_indices[i] = i;

                    KdTreeBuilder<scalar_t> builder(pts_b, leaf_size);
                    builder.build(initial_indices, 0);

                    tree_data[b].split_dim = std::move(builder.split_dim);
                    tree_data[b].left = std::move(builder.left);
                    tree_data[b].right = std::move(builder.right);
                    tree_data[b].indices = std::move(builder.final_indices);
                    tree_data[b].leaf_starts = std::move(builder.leaf_starts);
                    tree_data[b].leaf_counts = std::move(builder.leaf_counts);

                    // Store split_val in native dtype (no conversion)
                    typed_split_vals[b] = std::move(builder.split_val);
                }
            });

            // Copy to output tensors (still inside dispatch for correct scalar_t)
            // Find max sizes
            int64_t max_nodes = 0;
            int64_t max_leaves = 0;
            for (int64_t b = 0; b < B; ++b) {
                max_nodes = std::max(max_nodes, static_cast<int64_t>(tree_data[b].split_dim.size()));
                max_leaves = std::max(max_leaves, static_cast<int64_t>(tree_data[b].leaf_starts.size()));
            }

            // Allocate and copy (see Phase 3/4 below for tensor allocation)
            // Store typed vectors for Phase 4 copy
            for (int64_t b = 0; b < B; ++b) {
                split_val_storage[b] = std::make_shared<std::vector<scalar_t>>(
                    std::move(typed_split_vals[b])
                );
            }
        }
    );

    // Phase 2: Find max sizes (already computed inside dispatch above)
    // Re-compute here since we're outside the dispatch block
    int64_t max_nodes = 0;
    int64_t max_leaves = 0;
    for (int64_t b = 0; b < B; ++b) {
        max_nodes = std::max(max_nodes, static_cast<int64_t>(tree_data[b].split_dim.size()));
        max_leaves = std::max(max_leaves, static_cast<int64_t>(tree_data[b].leaf_starts.size()));
    }

    // Phase 3: Allocate output tensors with padding
    at::Tensor split_dim_out = at::full({B, max_nodes}, -1, points.options().dtype(at::kLong));
    at::Tensor split_val_out = at::zeros({B, max_nodes}, points.options());
    at::Tensor left_out = at::full({B, max_nodes}, -1, points.options().dtype(at::kLong));
    at::Tensor right_out = at::full({B, max_nodes}, -1, points.options().dtype(at::kLong));
    at::Tensor indices_out = at::empty({B, n}, points.options().dtype(at::kLong));
    at::Tensor leaf_starts_out = at::zeros({B, max_leaves}, points.options().dtype(at::kLong));
    at::Tensor leaf_counts_out = at::zeros({B, max_leaves}, points.options().dtype(at::kLong));

    // Phase 4: Copy data in parallel (split_val uses native dtype, no conversion)
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "copy_tree_data",
        [&]() {
            at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    auto& td = tree_data[b];
                    int64_t n_nodes = static_cast<int64_t>(td.split_dim.size());
                    int64_t n_leaves = static_cast<int64_t>(td.leaf_starts.size());

                    std::memcpy(split_dim_out[b].data_ptr<int64_t>(), td.split_dim.data(), n_nodes * sizeof(int64_t));
                    std::memcpy(left_out[b].data_ptr<int64_t>(), td.left.data(), n_nodes * sizeof(int64_t));
                    std::memcpy(right_out[b].data_ptr<int64_t>(), td.right.data(), n_nodes * sizeof(int64_t));
                    std::memcpy(indices_out[b].data_ptr<int64_t>(), td.indices.data(), n * sizeof(int64_t));
                    std::memcpy(leaf_starts_out[b].data_ptr<int64_t>(), td.leaf_starts.data(), n_leaves * sizeof(int64_t));
                    std::memcpy(leaf_counts_out[b].data_ptr<int64_t>(), td.leaf_counts.data(), n_leaves * sizeof(int64_t));

                    // Copy split_val directly (already in native dtype)
                    auto* typed_vals = static_cast<std::vector<scalar_t>*>(split_val_storage[b].get());
                    std::memcpy(split_val_out[b].data_ptr<scalar_t>(), typed_vals->data(), n_nodes * sizeof(scalar_t));
                }
            });
        }
    );

    return std::make_tuple(
        points_contig,
        split_dim_out,
        split_val_out,
        left_out,
        right_out,
        indices_out,
        leaf_starts_out,
        leaf_counts_out
    );
}

}  // namespace torchscience::cpu::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("kd_tree_build_batched", torchscience::cpu::space_partitioning::kd_tree_build_batched);
}
