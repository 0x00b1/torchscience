// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/space_partitioning/kd_tree.h"
#include "cpu/space_partitioning/k_nearest_neighbors.h"
#include "cpu/space_partitioning/range_search.h"
#include "cpu/space_partitioning/bvh.h"
#include "cpu/space_partitioning/octree.h"

// Meta backend
#include "meta/space_partitioning/kd_tree.h"
#include "meta/space_partitioning/k_nearest_neighbors.h"
#include "meta/space_partitioning/range_search.h"
#include "meta/space_partitioning/bvh.h"

// Autograd backend
#include "autograd/space_partitioning/k_nearest_neighbors.h"
#include "autograd/space_partitioning/range_search.h"
#include "autograd/space_partitioning/octree.h"

// Autocast backend
#include "autocast/space_partitioning/kd_tree.h"
#include "autocast/space_partitioning/k_nearest_neighbors.h"
#include "autocast/space_partitioning/range_search.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/space_partitioning/kd_tree.cuh"
#endif

#ifdef TORCHSCIENCE_OPTIX
#include "optix/space_partitioning/bvh.h"
#include "optix/geometry/ray_intersect.h"
#include "optix/geometry/ray_occluded.h"
#include "optix/geometry/closest_point.h"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // KD-tree
  m.def("kd_tree_build_batched(Tensor points, int leaf_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // k-nearest neighbors query
  m.def("k_nearest_neighbors(Tensor points, Tensor split_dim, Tensor split_val, Tensor left, Tensor right, Tensor indices, Tensor leaf_starts, Tensor leaf_counts, Tensor queries, int k, float p) -> (Tensor, Tensor)");

  // Range search query
  m.def("range_search(Tensor points, Tensor split_dim, Tensor split_val, Tensor left, Tensor right, Tensor indices, Tensor leaf_starts, Tensor leaf_counts, Tensor queries, float radius, float p) -> (Tensor, Tensor)");

  // BVH (Bounding Volume Hierarchy)
  m.def("bvh_build(Tensor vertices, Tensor faces) -> Tensor");
  m.def("bvh_destroy(int scene_handle) -> ()");
  m.def("bvh_ray_intersect(int scene_handle, Tensor origins, Tensor directions) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("bvh_closest_point(int scene_handle, Tensor query_points) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("bvh_ray_occluded(int scene_handle, Tensor origins, Tensor directions) -> Tensor");

  // Octree construction
  m.def("octree_build(Tensor points, Tensor data, int maximum_depth, float capacity_factor, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Octree queries
  m.def("octree_sample(Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor points, int maximum_depth, int interpolation, int? query_depth) -> (Tensor, Tensor)");
  m.def("octree_sample_backward(Tensor grad_output, Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor points, int maximum_depth, int interpolation, int? query_depth) -> (Tensor, Tensor)");

  // Octree ray marching
  m.def("octree_ray_marching(Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor origins, Tensor directions, int maximum_depth, float? step_size, int maximum_steps) -> (Tensor, Tensor, Tensor)");
  m.def("octree_ray_marching_backward(Tensor grad_positions, Tensor grad_data_out, Tensor mask, Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor origins, Tensor directions, int maximum_depth, float? step_size, int maximum_steps) -> (Tensor, Tensor, Tensor)");

  // Octree neighbor finding
  m.def("octree_neighbors(Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor query_codes, int connectivity) -> (Tensor, Tensor)");

  // Octree dynamic updates
  m.def("octree_insert(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor new_points, Tensor new_data, int depth, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("octree_remove(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor remove_codes, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("octree_subdivide(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor subdivide_codes, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("octree_merge(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor merge_codes, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
