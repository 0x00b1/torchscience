#pragma once

#include <cmath>
#include <limits>
#include <atomic>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
void floyd_warshall_single(
    scalar_t* dist,
    int64_t* pred,
    int64_t N,
    bool directed,
    bool& has_negative_cycle
) {
    const scalar_t inf = std::numeric_limits<scalar_t>::infinity();

    // If undirected, symmetrize by taking element-wise minimum
    if (!directed) {
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = i + 1; j < N; ++j) {
                scalar_t min_val = std::min(dist[i * N + j], dist[j * N + i]);
                dist[i * N + j] = min_val;
                dist[j * N + i] = min_val;
            }
        }
    }

    // Initialize predecessors:
    // pred[i,j] = i if edge exists (dist[i,j] < inf), else -1
    // pred[i,i] = -1 (no predecessor for self)
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            if (i == j) {
                pred[i * N + j] = -1;
                dist[i * N + i] = scalar_t(0);  // Diagonal is always 0
            } else if (dist[i * N + j] < inf) {
                pred[i * N + j] = i;
            } else {
                pred[i * N + j] = -1;
            }
        }
    }

    // Floyd-Warshall main loop
    for (int64_t k = 0; k < N; ++k) {
        for (int64_t i = 0; i < N; ++i) {
            scalar_t dist_ik = dist[i * N + k];
            if (dist_ik >= inf) continue;  // Skip if no path i->k

            for (int64_t j = 0; j < N; ++j) {
                scalar_t dist_kj = dist[k * N + j];
                if (dist_kj >= inf) continue;  // Skip if no path k->j

                scalar_t new_dist = dist_ik + dist_kj;
                if (new_dist < dist[i * N + j]) {
                    dist[i * N + j] = new_dist;
                    pred[i * N + j] = pred[k * N + j];
                }
            }
        }
    }

    // Check for negative cycles (diagonal < 0)
    for (int64_t i = 0; i < N; ++i) {
        if (dist[i * N + i] < scalar_t(0)) {
            has_negative_cycle = true;
            return;
        }
    }
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor, bool> floyd_warshall(
    const at::Tensor& input,
    bool directed
) {
    TORCH_CHECK(
        input.dim() >= 2,
        "floyd_warshall: input must be at least 2D, got ", input.dim(), "D"
    );
    TORCH_CHECK(
        input.size(-1) == input.size(-2),
        "floyd_warshall: last two dimensions must be equal, got ",
        input.size(-2), " x ", input.size(-1)
    );
    TORCH_CHECK(
        at::isFloatingType(input.scalar_type()),
        "floyd_warshall: input must be floating-point, got ", input.scalar_type()
    );

    // Handle sparse input
    at::Tensor dense_input = input.is_sparse() ? input.to_dense() : input;

    // Make contiguous copy for in-place modification
    at::Tensor distances = dense_input.clone().contiguous();

    int64_t N = distances.size(-1);
    int64_t batch_size = distances.numel() / (N * N);

    at::Tensor predecessors = at::empty(
        distances.sizes(),
        distances.options().dtype(at::kLong)
    ).contiguous();

    // Handle empty graph
    if (N == 0) {
        return std::make_tuple(distances, predecessors, false);
    }

    bool has_negative_cycle = false;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16,
        distances.scalar_type(),
        "floyd_warshall_cpu",
        [&] {
            scalar_t* dist_ptr = distances.data_ptr<scalar_t>();
            int64_t* pred_ptr = predecessors.data_ptr<int64_t>();

            // Process batches in parallel
            std::atomic<bool> found_negative_cycle{false};

            at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
                for (int64_t b = start; b < end; ++b) {
                    if (found_negative_cycle.load()) return;

                    bool local_neg_cycle = false;
                    floyd_warshall_single<scalar_t>(
                        dist_ptr + b * N * N,
                        pred_ptr + b * N * N,
                        N,
                        directed,
                        local_neg_cycle
                    );

                    if (local_neg_cycle) {
                        found_negative_cycle.store(true);
                    }
                }
            });

            has_negative_cycle = found_negative_cycle.load();
        }
    );

    return std::make_tuple(distances, predecessors, has_negative_cycle);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("floyd_warshall", &torchscience::cpu::graph_theory::floyd_warshall);
}
