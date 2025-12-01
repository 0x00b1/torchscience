#pragma once

#include <cmath>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::space_partitioning {

namespace {

template <typename scalar_t>
struct RangeSearcher {
    const scalar_t* points;
    const int64_t* split_dim;
    const scalar_t* split_val;
    const int64_t* left_child;
    const int64_t* right_child;
    const int64_t* indices;
    const int64_t* leaf_starts;
    const int64_t* leaf_counts;
    int64_t n_nodes;
    int64_t d;
    scalar_t p;
    scalar_t radius;

    void search(
        const scalar_t* query,
        std::vector<std::pair<scalar_t, int64_t>>& results
    ) const {
        if (n_nodes == 0) return;
        search_node(0, query, results);
    }

    void search_node(
        int64_t node,
        const scalar_t* query,
        std::vector<std::pair<scalar_t, int64_t>>& results
    ) const {
        if (node < 0 || node >= n_nodes) return;

        int64_t sd = split_dim[node];

        // Leaf node - check all points
        if (sd == -1) {
            // Find leaf index
            int64_t leaf_idx = 0;
            for (int64_t i = 0; i < node; ++i) {
                if (split_dim[i] == -1) leaf_idx++;
            }

            int64_t start = leaf_starts[leaf_idx];
            int64_t count = leaf_counts[leaf_idx];

            for (int64_t i = 0; i < count; ++i) {
                int64_t pt_idx = indices[start + i];

                // Compute distance
                scalar_t dist = scalar_t(0);
                const scalar_t* pt = points + pt_idx * d;

                if (p == scalar_t(2)) {
                    for (int64_t j = 0; j < d; ++j) {
                        scalar_t diff = query[j] - pt[j];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                } else if (p == scalar_t(1)) {
                    for (int64_t j = 0; j < d; ++j) {
                        dist += std::abs(query[j] - pt[j]);
                    }
                } else {
                    for (int64_t j = 0; j < d; ++j) {
                        dist += std::pow(std::abs(query[j] - pt[j]), p);
                    }
                    dist = std::pow(dist, scalar_t(1) / p);
                }

                if (dist <= radius) {
                    results.push_back({dist, pt_idx});
                }
            }
            return;
        }

        // Internal node
        scalar_t sv = split_val[node];
        scalar_t query_val = query[sd];
        scalar_t diff = query_val - sv;

        // Visit children that could contain points within radius
        // Left child: points with coord < sv
        if (diff - radius < 0) {
            search_node(left_child[node], query, results);
        }
        // Right child: points with coord >= sv
        if (diff + radius >= 0) {
            search_node(right_child[node], query, results);
        }
    }
};

}  // anonymous namespace

// Returns nested tensors for indices and distances
inline std::tuple<at::Tensor, at::Tensor> range_search(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    double radius,
    double p
) {
    TORCH_CHECK(points.dim() == 2, "range_search: points must be 2D");
    TORCH_CHECK(queries.dim() == 2, "range_search: queries must be 2D");
    TORCH_CHECK(points.size(1) == queries.size(1), "range_search: dimension mismatch");
    TORCH_CHECK(radius >= 0, "range_search: radius must be non-negative");

    int64_t d = points.size(1);
    int64_t m = queries.size(0);
    int64_t n_nodes = split_dim.size(0);

    at::Tensor pts_contig = points.contiguous();
    at::Tensor queries_contig = queries.contiguous();
    at::Tensor split_dim_contig = split_dim.contiguous();
    at::Tensor split_val_contig = split_val.contiguous();
    at::Tensor left_contig = left.contiguous();
    at::Tensor right_contig = right.contiguous();
    at::Tensor indices_contig = indices.contiguous();
    at::Tensor leaf_starts_contig = leaf_starts.contiguous();
    at::Tensor leaf_counts_contig = leaf_counts.contiguous();

    // Collect results per query
    std::vector<std::vector<std::pair<double, int64_t>>> all_results(m);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "range_search_cpu",
        [&]() {
            RangeSearcher<scalar_t> searcher{
                pts_contig.data_ptr<scalar_t>(),
                split_dim_contig.data_ptr<int64_t>(),
                split_val_contig.data_ptr<scalar_t>(),
                left_contig.data_ptr<int64_t>(),
                right_contig.data_ptr<int64_t>(),
                indices_contig.data_ptr<int64_t>(),
                leaf_starts_contig.data_ptr<int64_t>(),
                leaf_counts_contig.data_ptr<int64_t>(),
                n_nodes,
                d,
                static_cast<scalar_t>(p),
                static_cast<scalar_t>(radius)
            };

            const scalar_t* queries_ptr = queries_contig.data_ptr<scalar_t>();

            at::parallel_for(0, m, 0, [&](int64_t begin, int64_t end) {
                for (int64_t q_idx = begin; q_idx < end; ++q_idx) {
                    const scalar_t* query = queries_ptr + q_idx * d;

                    std::vector<std::pair<scalar_t, int64_t>> results;
                    searcher.search(query, results);

                    // Sort by distance
                    std::sort(results.begin(), results.end());

                    // Convert to double for storage
                    for (const auto& [dist, idx] : results) {
                        all_results[q_idx].push_back({static_cast<double>(dist), idx});
                    }
                }
            });
        }
    );

    // Create nested tensors
    std::vector<at::Tensor> idx_tensors(m);
    std::vector<at::Tensor> dist_tensors(m);

    for (int64_t i = 0; i < m; ++i) {
        int64_t count = static_cast<int64_t>(all_results[i].size());

        if (count > 0) {
            idx_tensors[i] = at::empty({count}, queries.options().dtype(at::kLong));
            dist_tensors[i] = at::empty({count}, queries.options());

            int64_t* idx_ptr = idx_tensors[i].data_ptr<int64_t>();
            for (int64_t j = 0; j < count; ++j) {
                idx_ptr[j] = all_results[i][j].second;
            }

            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kBFloat16, at::kHalf,
                queries.scalar_type(),
                "range_search_copy_dist",
                [&]() {
                    scalar_t* dist_ptr = dist_tensors[i].data_ptr<scalar_t>();
                    for (int64_t j = 0; j < count; ++j) {
                        dist_ptr[j] = static_cast<scalar_t>(all_results[i][j].first);
                    }
                }
            );
        } else {
            idx_tensors[i] = at::empty({0}, queries.options().dtype(at::kLong));
            dist_tensors[i] = at::empty({0}, queries.options());
        }
    }

    // Create nested tensors from list of tensors
    at::Tensor nested_indices = at::_nested_tensor_from_tensor_list(idx_tensors);
    at::Tensor nested_distances = at::_nested_tensor_from_tensor_list(dist_tensors);

    return std::make_tuple(nested_indices, nested_distances);
}

}  // namespace torchscience::cpu::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("range_search", &torchscience::cpu::space_partitioning::range_search);
}
