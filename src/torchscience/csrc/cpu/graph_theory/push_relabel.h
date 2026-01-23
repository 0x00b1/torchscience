#pragma once

#include <algorithm>
#include <limits>
#include <queue>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor> push_relabel_impl(
    const scalar_t* capacity_ptr,
    int64_t N,
    int64_t source,
    int64_t sink,
    const at::TensorOptions& options
) {
  // Handle source == sink case
  if (source == sink) {
    at::Tensor max_flow = at::zeros({}, options);
    at::Tensor flow = at::zeros({N, N}, options);
    return std::make_tuple(max_flow, flow);
  }

  // Initialize capacity and flow matrices
  std::vector<std::vector<scalar_t>> capacity(N, std::vector<scalar_t>(N, scalar_t(0)));
  std::vector<std::vector<scalar_t>> flow_values(N, std::vector<scalar_t>(N, scalar_t(0)));

  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      capacity[i][j] = capacity_ptr[i * N + j];
    }
  }

  // Initialize heights and excess
  // h[source] = N, h[others] = 0
  std::vector<int64_t> height(N, 0);
  std::vector<scalar_t> excess(N, scalar_t(0));

  height[source] = N;

  // Initialize preflow: saturate all edges from source
  for (int64_t v = 0; v < N; ++v) {
    if (capacity[source][v] > scalar_t(0)) {
      flow_values[source][v] = capacity[source][v];
      excess[v] = capacity[source][v];
      excess[source] -= capacity[source][v];
    }
  }

  // Maintain list of active nodes (excess > 0, not source, not sink)
  // Use a queue for FIFO selection of active nodes
  std::vector<bool> in_queue(N, false);
  std::queue<int64_t> active_queue;

  for (int64_t v = 0; v < N; ++v) {
    if (v != source && v != sink && excess[v] > scalar_t(0)) {
      active_queue.push(v);
      in_queue[v] = true;
    }
  }

  // Helper lambda to compute residual capacity
  auto residual = [&](int64_t u, int64_t v) -> scalar_t {
    return capacity[u][v] - flow_values[u][v] + flow_values[v][u];
  };

  // Push operation
  auto push = [&](int64_t u, int64_t v) {
    // Compute actual push amount
    scalar_t res_cap = residual(u, v);
    scalar_t push_amount = std::min(excess[u], res_cap);

    if (push_amount <= scalar_t(0)) {
      return;
    }

    // Update flow: prefer forward edge, then reduce reverse
    if (flow_values[v][u] > scalar_t(0)) {
      // First cancel flow on reverse edge
      scalar_t cancel = std::min(push_amount, flow_values[v][u]);
      flow_values[v][u] -= cancel;
      push_amount -= cancel;
      excess[u] -= cancel;
      excess[v] += cancel;
    }
    if (push_amount > scalar_t(0)) {
      flow_values[u][v] += push_amount;
      excess[u] -= push_amount;
      excess[v] += push_amount;
    }

    // Add v to active queue if it became active
    if (v != source && v != sink && excess[v] > scalar_t(0) && !in_queue[v]) {
      active_queue.push(v);
      in_queue[v] = true;
    }
  };

  // Relabel operation
  auto relabel = [&](int64_t u) {
    int64_t min_height = std::numeric_limits<int64_t>::max();
    for (int64_t v = 0; v < N; ++v) {
      if (residual(u, v) > scalar_t(0)) {
        min_height = std::min(min_height, height[v]);
      }
    }
    if (min_height < std::numeric_limits<int64_t>::max()) {
      height[u] = min_height + 1;
    }
  };

  // Discharge operation: push as much as possible from u, relabel if needed
  auto discharge = [&](int64_t u) {
    while (excess[u] > scalar_t(0)) {
      bool pushed = false;
      for (int64_t v = 0; v < N; ++v) {
        if (residual(u, v) > scalar_t(0) && height[u] == height[v] + 1) {
          push(u, v);
          pushed = true;
          if (excess[u] <= scalar_t(0)) {
            break;
          }
        }
      }
      if (!pushed) {
        relabel(u);
        // Safety check: if height exceeds 2*N, something is wrong
        if (height[u] > 2 * N) {
          break;
        }
      }
    }
  };

  // Main loop: process active nodes
  // Use relabel-to-front heuristic could be more efficient, but FIFO is simpler
  while (!active_queue.empty()) {
    int64_t u = active_queue.front();
    active_queue.pop();
    in_queue[u] = false;

    if (excess[u] > scalar_t(0)) {
      discharge(u);
      // If still has excess after discharge, re-add to queue
      if (excess[u] > scalar_t(0) && u != source && u != sink) {
        active_queue.push(u);
        in_queue[u] = true;
      }
    }
  }

  // Max flow is the excess at sink
  scalar_t total_flow = excess[sink];

  // Convert to tensors
  at::Tensor max_flow_tensor = at::empty({}, options);
  at::Tensor flow_tensor = at::empty({N, N}, options);

  max_flow_tensor.fill_(total_flow);

  auto flow_ptr = flow_tensor.data_ptr<scalar_t>();
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      flow_ptr[i * N + j] = flow_values[i][j];
    }
  }

  return std::make_tuple(max_flow_tensor, flow_tensor);
}

}  // anonymous namespace

inline std::tuple<at::Tensor, at::Tensor> push_relabel(
    const at::Tensor& capacity,
    int64_t source,
    int64_t sink
) {
  TORCH_CHECK(
      capacity.dim() == 2,
      "push_relabel: capacity must be 2D, got ", capacity.dim(), "D"
  );
  TORCH_CHECK(
      capacity.size(0) == capacity.size(1),
      "push_relabel: capacity must be square, got ",
      capacity.size(0), " x ", capacity.size(1)
  );

  int64_t N = capacity.size(0);

  TORCH_CHECK(
      source >= 0 && source < N,
      "push_relabel: source must be in [0, ", N - 1, "], got ", source
  );
  TORCH_CHECK(
      sink >= 0 && sink < N,
      "push_relabel: sink must be in [0, ", N - 1, "], got ", sink
  );

  // Handle sparse input
  at::Tensor dense_capacity = capacity.is_sparse() ? capacity.to_dense() : capacity;
  dense_capacity = dense_capacity.contiguous();

  // Handle empty graph
  if (N == 0) {
    return std::make_tuple(
        at::zeros({}, dense_capacity.options()),
        at::empty({0, 0}, dense_capacity.options())
    );
  }

  at::Tensor max_flow, flow;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_capacity.scalar_type(),
      "push_relabel_cpu",
      [&] {
        const scalar_t* cap_ptr = dense_capacity.data_ptr<scalar_t>();
        std::tie(max_flow, flow) = push_relabel_impl<scalar_t>(
            cap_ptr, N, source, sink, dense_capacity.options()
        );
      }
  );

  return std::make_tuple(max_flow, flow);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("push_relabel", &torchscience::cpu::graph_theory::push_relabel);
}
