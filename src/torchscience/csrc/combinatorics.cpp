// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/combinatorics.h"

// Meta backend
#include "meta/combinatorics.h"

// Autograd backend
#include "autograd/combinatorics.h"

// Autocast backend
#include "autocast/combinatorics.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Binomial coefficient
  m.def("binomial_coefficient(Tensor n, Tensor k) -> Tensor");
  m.def("binomial_coefficient_backward(Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor)");
  m.def("binomial_coefficient_backward_backward(Tensor gg_n, Tensor gg_k, Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor, Tensor)");
}
