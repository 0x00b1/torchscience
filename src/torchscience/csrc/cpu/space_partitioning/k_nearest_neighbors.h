// src/torchscience/csrc/cpu/space_partitioning/k_nearest_neighbors.h
#pragma once

#include <cmath>
#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <torch/library.h>

namespace torchscience::cpu::space_partitioning {

namespace {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T minkowski_distance(const T* x, const T* y, int64_t d, T p) {
    T sum = T(0);
    if (p == T(2)) {
        for (int64_t i = 0; i < d; ++i) {
            T diff = x[i] - y[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    } else if (p == T(1)) {
        for (int64_t i = 0; i < d; ++i) {
            sum += std::abs(x[i] - y[i]);
        }
        return sum;
    } else {
        for (int64_t i = 0; i < d; ++i) {
            sum += std::pow(std::abs(x[i] - y[i]), p);
        }
        return std::pow(sum, T(1) / p);
    }
}

template <typename scalar_t>
struct KNNSearcher {
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

    // Max-heap: (distance, index) - largest distance at top
    using HeapItem = std::pair<scalar_t, int64_t>;
    struct MaxHeapCmp {
        bool operator()(const HeapItem& a, const HeapItem& b) {
            return a.first < b.first;
        }
    };

    void search(
        const scalar_t* query,
        int64_t k,
        std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCmp>& heap
    ) const {
        if (n_nodes == 0) return;
        search_node(0, query, k, heap);
    }

    void search_node(
        int64_t node,
        const scalar_t* query,
        int64_t k,
        std::priority_queue<HeapItem, std::vector<HeapItem>, MaxHeapCmp>& heap
    ) const {
        if (node < 0 || node >= n_nodes) return;

        int64_t sd = split_dim[node];

        // Leaf node
        if (sd == -1) {
            // Find leaf index by counting leaves before this node
            int64_t leaf_idx = 0;
            for (int64_t i = 0; i < node; ++i) {
                if (split_dim[i] == -1) leaf_idx++;
            }

            int64_t start = leaf_starts[leaf_idx];
            int64_t count = leaf_counts[leaf_idx];

            for (int64_t i = 0; i < count; ++i) {
                int64_t pt_idx = indices[start + i];
                scalar_t dist = minkowski_distance(
                    query, points + pt_idx * d, d, p
                );

                if (static_cast<int64_t>(heap.size()) < k) {
                    heap.push({dist, pt_idx});
                } else if (dist < heap.top().first) {
                    heap.pop();
                    heap.push({dist, pt_idx});
                }
            }
            return;
        }

        // Internal node
        scalar_t sv = split_val[node];
        scalar_t query_val = query[sd];
        scalar_t diff = query_val - sv;

        // Determine which child to visit first
        int64_t first_child = diff < 0 ? left_child[node] : right_child[node];
        int64_t second_child = diff < 0 ? right_child[node] : left_child[node];

        // Visit closer child first
        search_node(first_child, query, k, heap);

        // Prune: only visit other child if it could contain closer points
        scalar_t plane_dist = std::abs(diff);
        if (static_cast<int64_t>(heap.size()) < k || plane_dist < heap.top().first) {
            search_node(second_child, query, k, heap);
        }
    }
};

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor> k_nearest_neighbors(
    const at::Tensor& points,
    const at::Tensor& split_dim,
    const at::Tensor& split_val,
    const at::Tensor& left,
    const at::Tensor& right,
    const at::Tensor& indices,
    const at::Tensor& leaf_starts,
    const at::Tensor& leaf_counts,
    const at::Tensor& queries,
    int64_t k,
    double p
) {
    TORCH_CHECK(points.dim() == 2, "k_nearest_neighbors: points must be 2D");
    TORCH_CHECK(queries.dim() == 2, "k_nearest_neighbors: queries must be 2D");
    TORCH_CHECK(points.size(1) == queries.size(1), "k_nearest_neighbors: dimension mismatch");

    int64_t n = points.size(0);
    int64_t d = points.size(1);
    int64_t m = queries.size(0);
    int64_t n_nodes = split_dim.size(0);

    TORCH_CHECK(k > 0 && k <= n, "k_nearest_neighbors: k must be in [1, n]");

    at::Tensor pts_contig = points.contiguous();
    at::Tensor queries_contig = queries.contiguous();
    at::Tensor split_dim_contig = split_dim.contiguous();
    at::Tensor split_val_contig = split_val.contiguous();
    at::Tensor left_contig = left.contiguous();
    at::Tensor right_contig = right.contiguous();
    at::Tensor indices_contig = indices.contiguous();
    at::Tensor leaf_starts_contig = leaf_starts.contiguous();
    at::Tensor leaf_counts_contig = leaf_counts.contiguous();

    at::Tensor result_indices = at::empty({m, k}, queries.options().dtype(at::kLong));
    at::Tensor result_distances = at::empty({m, k}, queries.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf,
        points.scalar_type(),
        "k_nearest_neighbors_cpu",
        [&]() {
            KNNSearcher<scalar_t> searcher{
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
                static_cast<scalar_t>(p)
            };

            const scalar_t* queries_ptr = queries_contig.data_ptr<scalar_t>();
            int64_t* result_idx_ptr = result_indices.data_ptr<int64_t>();
            scalar_t* result_dist_ptr = result_distances.data_ptr<scalar_t>();

            at::parallel_for(0, m, 0, [&](int64_t begin, int64_t end) {
                for (int64_t q_idx = begin; q_idx < end; ++q_idx) {
                    const scalar_t* query = queries_ptr + q_idx * d;

                    std::priority_queue<
                        std::pair<scalar_t, int64_t>,
                        std::vector<std::pair<scalar_t, int64_t>>,
                        typename KNNSearcher<scalar_t>::MaxHeapCmp
                    > heap;

                    searcher.search(query, k, heap);

                    // Extract results (heap gives largest first, reverse for ascending)
                    std::vector<std::pair<scalar_t, int64_t>> results;
                    while (!heap.empty()) {
                        results.push_back(heap.top());
                        heap.pop();
                    }
                    std::reverse(results.begin(), results.end());

                    // Pad if fewer than k points
                    while (static_cast<int64_t>(results.size()) < k) {
                        results.push_back({std::numeric_limits<scalar_t>::infinity(), -1});
                    }

                    for (int64_t i = 0; i < k; ++i) {
                        result_idx_ptr[q_idx * k + i] = results[i].second;
                        result_dist_ptr[q_idx * k + i] = results[i].first;
                    }
                }
            });
        }
    );

    return std::make_tuple(result_indices, result_distances);
}

}  // namespace torchscience::cpu::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("k_nearest_neighbors", &torchscience::cpu::space_partitioning::k_nearest_neighbors);
}
