// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/graph_theory/floyd_warshall.h"
#include "cpu/graph_theory/connected_components.h"
#include "cpu/graph_theory/dijkstra.h"
#include "cpu/graph_theory/bellman_ford.h"
#include "cpu/graph_theory/minimum_spanning_tree.h"
#include "cpu/graph_theory/maximum_bipartite_matching.h"
#include "cpu/graph_theory/closeness_centrality.h"
#include "cpu/graph_theory/katz_centrality.h"
#include "cpu/graph_theory/eigenvector_centrality.h"
#include "cpu/graph_theory/betweenness_centrality.h"
#include "cpu/graph_theory/topological_sort.h"
#include "cpu/graph_theory/breadth_first_search.h"
#include "cpu/graph_theory/depth_first_search.h"
#include "cpu/graph_theory/dag_shortest_paths.h"
#include "cpu/graph_theory/edmonds_karp.h"
#include "cpu/graph_theory/push_relabel.h"
#include "cpu/graph_theory/minimum_cut.h"
#include "cpu/graph_theory/min_cost_max_flow.h"

// Meta backend
#include "meta/graph_theory/floyd_warshall.h"
#include "meta/graph_theory/connected_components.h"
#include "meta/graph_theory/dijkstra.h"
#include "meta/graph_theory/bellman_ford.h"
#include "meta/graph_theory/minimum_spanning_tree.h"
#include "meta/graph_theory/maximum_bipartite_matching.h"
#include "meta/graph_theory/closeness_centrality.h"
#include "meta/graph_theory/katz_centrality.h"
#include "meta/graph_theory/eigenvector_centrality.h"
#include "meta/graph_theory/betweenness_centrality.h"
#include "meta/graph_theory/topological_sort.h"
#include "meta/graph_theory/breadth_first_search.h"
#include "meta/graph_theory/depth_first_search.h"
#include "meta/graph_theory/dag_shortest_paths.h"
#include "meta/graph_theory/edmonds_karp.h"
#include "meta/graph_theory/push_relabel.h"
#include "meta/graph_theory/minimum_cut.h"
#include "meta/graph_theory/min_cost_max_flow.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/graph_theory/floyd_warshall.cu"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Shortest paths
  m.def("floyd_warshall(Tensor input, bool directed) -> (Tensor, Tensor, bool)");
  m.def("dijkstra(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor)");
  m.def("bellman_ford(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor, bool)");
  m.def("dag_shortest_paths(Tensor adjacency, int source) -> (Tensor, Tensor)");

  // Connectivity
  m.def("connected_components(Tensor adjacency, bool directed, str connection) -> (int, Tensor)");

  // Minimum spanning tree
  m.def("minimum_spanning_tree(Tensor adjacency) -> (Tensor, Tensor)");

  // Matching
  m.def("maximum_bipartite_matching(Tensor biadjacency) -> (Tensor, Tensor, Tensor)");

  // Centrality measures
  m.def("closeness_centrality(Tensor adjacency, bool normalized) -> Tensor");
  m.def("closeness_centrality_backward(Tensor grad, Tensor adjacency, Tensor distances, bool normalized) -> Tensor");
  m.def("katz_centrality(Tensor adjacency, float alpha, float beta, bool normalized) -> Tensor");
  m.def("eigenvector_centrality(Tensor adjacency) -> Tensor");
  m.def("betweenness_centrality(Tensor adjacency, bool normalized) -> Tensor");

  // Graph traversal
  m.def("topological_sort(Tensor adjacency) -> Tensor");
  m.def("breadth_first_search(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor)");
  m.def("depth_first_search(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor, Tensor)");

  // Network flow
  m.def("edmonds_karp(Tensor capacity, int source, int sink) -> (Tensor, Tensor)");
  m.def("push_relabel(Tensor capacity, int source, int sink) -> (Tensor, Tensor)");
  m.def("minimum_cut(Tensor capacity, int source, int sink) -> (Tensor, Tensor, Tensor)");
  m.def("min_cost_max_flow(Tensor capacity, Tensor cost, int source, int sink) -> (Tensor, Tensor, Tensor)");
}
