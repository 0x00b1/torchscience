// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/distance/minkowski_distance.h"
#include "cpu/distance/hellinger_distance.h"
#include "cpu/distance/total_variation_distance.h"
#include "cpu/distance/bhattacharyya_distance.h"

// Meta backend
#include "meta/distance/minkowski_distance.h"
#include "meta/distance/hellinger_distance.h"
#include "meta/distance/total_variation_distance.h"
#include "meta/distance/bhattacharyya_distance.h"

// Autograd backend
#include "autograd/distance/minkowski_distance.h"
#include "autograd/distance/hellinger_distance.h"
#include "autograd/distance/total_variation_distance.h"
#include "autograd/distance/bhattacharyya_distance.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Minkowski distance
  m.def("minkowski_distance(Tensor x, Tensor y, float p, Tensor? weight) -> Tensor");
  m.def("minkowski_distance_backward(Tensor grad_output, Tensor x, Tensor y, float p, Tensor? weight, Tensor dist_output) -> (Tensor, Tensor, Tensor)");

  // Hellinger distance
  m.def("hellinger_distance(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  m.def("hellinger_distance_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");

  // Total variation distance
  m.def("total_variation_distance(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  m.def("total_variation_distance_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");

  // Bhattacharyya distance
  m.def("bhattacharyya_distance(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  m.def("bhattacharyya_distance_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");
}
