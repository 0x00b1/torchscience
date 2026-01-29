// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/test/sum_squares.h"

// Meta backend
#include "meta/test/sum_squares.h"

// Autograd backend
#include "autograd/test/sum_squares.h"

// Autocast backend
#include "autocast/test/sum_squares.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Test operators (for validating reduction macros)
  m.def("sum_squares(Tensor input, int[]? dim, bool keepdim) -> Tensor");
  m.def("sum_squares_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> Tensor");
  m.def("sum_squares_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> (Tensor, Tensor)");
}
