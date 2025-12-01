#pragma once

#include <algorithm>
#include <stack>
#include <vector>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::cpu::graph_theory {

namespace {

// Union-Find data structure for weak connectivity
class UnionFind {
 public:
  explicit UnionFind(int64_t n) : parent_(n), rank_(n, 0) {
    for (int64_t i = 0; i < n; ++i) {
      parent_[i] = i;
    }
  }

  int64_t find(int64_t x) {
    if (parent_[x] != x) {
      parent_[x] = find(parent_[x]);  // Path compression
    }
    return parent_[x];
  }

  void unite(int64_t x, int64_t y) {
    int64_t px = find(x);
    int64_t py = find(y);
    if (px == py) return;

    // Union by rank
    if (rank_[px] < rank_[py]) {
      parent_[px] = py;
    } else if (rank_[px] > rank_[py]) {
      parent_[py] = px;
    } else {
      parent_[py] = px;
      rank_[px]++;
    }
  }

 private:
  std::vector<int64_t> parent_;
  std::vector<int64_t> rank_;
};

// Weak connected components using Union-Find
template <typename scalar_t>
int64_t weak_connected_components(
    const scalar_t* adj,
    int64_t* labels,
    int64_t N
) {
  UnionFind uf(N);

  // Unite nodes connected by edges (in either direction)
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      if (adj[i * N + j] != scalar_t(0) || adj[j * N + i] != scalar_t(0)) {
        uf.unite(i, j);
      }
    }
  }

  // Assign component labels
  std::vector<int64_t> root_to_label(N, -1);
  int64_t n_components = 0;

  for (int64_t i = 0; i < N; ++i) {
    int64_t root = uf.find(i);
    if (root_to_label[root] == -1) {
      root_to_label[root] = n_components++;
    }
    labels[i] = root_to_label[root];
  }

  return n_components;
}

// Tarjan's algorithm for strongly connected components
template <typename scalar_t>
int64_t strong_connected_components(
    const scalar_t* adj,
    int64_t* labels,
    int64_t N
) {
  std::vector<int64_t> index(N, -1);
  std::vector<int64_t> lowlink(N, -1);
  std::vector<bool> on_stack(N, false);
  std::stack<int64_t> S;
  int64_t current_index = 0;
  int64_t n_components = 0;

  // Initialize labels to -1 (not assigned)
  for (int64_t i = 0; i < N; ++i) {
    labels[i] = -1;
  }

  // Build adjacency list for efficiency
  std::vector<std::vector<int64_t>> adj_list(N);
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      if (adj[i * N + j] != scalar_t(0)) {
        adj_list[i].push_back(j);
      }
    }
  }

  // Iterative Tarjan's algorithm (to avoid stack overflow)
  struct Frame {
    int64_t v;
    int64_t neighbor_idx;
    bool returned;
  };

  for (int64_t start = 0; start < N; ++start) {
    if (index[start] != -1) continue;

    std::stack<Frame> call_stack;
    call_stack.push({start, 0, false});

    while (!call_stack.empty()) {
      Frame& frame = call_stack.top();
      int64_t v = frame.v;

      if (!frame.returned && index[v] == -1) {
        // First visit to v
        index[v] = current_index;
        lowlink[v] = current_index;
        current_index++;
        S.push(v);
        on_stack[v] = true;
      }

      // Process neighbors
      bool pushed_child = false;
      while (frame.neighbor_idx < static_cast<int64_t>(adj_list[v].size())) {
        int64_t w = adj_list[v][frame.neighbor_idx];
        frame.neighbor_idx++;

        if (index[w] == -1) {
          // Recurse on w
          call_stack.push({w, 0, false});
          pushed_child = true;
          break;
        } else if (on_stack[w]) {
          lowlink[v] = std::min(lowlink[v], index[w]);
        }
      }

      if (pushed_child) continue;

      // After processing all neighbors
      if (frame.returned) {
        // Return from recursive call - update lowlink
        // The child we returned from is the last one we visited
        // We need to update based on child's lowlink
        // This is handled implicitly by our iterative structure
      }

      // Check if v is a root of an SCC
      if (lowlink[v] == index[v]) {
        // Pop nodes until we get v
        while (true) {
          int64_t w = S.top();
          S.pop();
          on_stack[w] = false;
          labels[w] = n_components;
          if (w == v) break;
        }
        n_components++;
      }

      call_stack.pop();

      // Update parent's lowlink
      if (!call_stack.empty()) {
        Frame& parent = call_stack.top();
        lowlink[parent.v] = std::min(lowlink[parent.v], lowlink[v]);
        parent.returned = true;
      }
    }
  }

  return n_components;
}

}  // anonymous namespace

inline std::tuple<int64_t, at::Tensor> connected_components(
    const at::Tensor& adjacency,
    bool directed,
    c10::string_view connection
) {
  TORCH_CHECK(
      adjacency.dim() >= 2,
      "connected_components: adjacency must be at least 2D, got ",
      adjacency.dim(), "D"
  );
  TORCH_CHECK(
      adjacency.size(-1) == adjacency.size(-2),
      "connected_components: adjacency must be square, got ",
      adjacency.size(-2), " x ", adjacency.size(-1)
  );
  TORCH_CHECK(
      connection == "weak" || connection == "strong",
      "connected_components: connection must be 'weak' or 'strong', got '",
      std::string(connection), "'"
  );
  TORCH_CHECK(
      !(connection == "strong" && !directed),
      "connected_components: strong connectivity requires directed=True"
  );

  // Handle sparse input
  at::Tensor dense_adj = adjacency.is_sparse() ? adjacency.to_dense() : adjacency;
  dense_adj = dense_adj.contiguous();

  int64_t N = dense_adj.size(-1);
  int64_t batch_size = dense_adj.numel() / (N * N);

  at::Tensor labels = at::empty(
      dense_adj.sizes().slice(0, dense_adj.dim() - 1),
      dense_adj.options().dtype(at::kLong)
  ).contiguous();

  // Handle empty graph
  if (N == 0) {
    return std::make_tuple(0, labels);
  }

  // For now, only support single graph (no batching)
  // Batching would require returning different n_components per batch
  TORCH_CHECK(
      batch_size == 1,
      "connected_components: batching not yet supported, got batch_size=",
      batch_size
  );

  int64_t n_components = 0;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16,
      dense_adj.scalar_type(),
      "connected_components_cpu",
      [&] {
        const scalar_t* adj_ptr = dense_adj.data_ptr<scalar_t>();
        int64_t* labels_ptr = labels.data_ptr<int64_t>();

        if (connection == "weak") {
          n_components = weak_connected_components<scalar_t>(
              adj_ptr, labels_ptr, N
          );
        } else {  // strong
          n_components = strong_connected_components<scalar_t>(
              adj_ptr, labels_ptr, N
          );
        }
      }
  );

  return std::make_tuple(n_components, labels);
}

}  // namespace torchscience::cpu::graph_theory

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("connected_components", &torchscience::cpu::graph_theory::connected_components);
}
