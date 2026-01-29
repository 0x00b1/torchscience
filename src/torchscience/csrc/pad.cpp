// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/pad/pad.h"

// Meta backend
#include "meta/pad/pad.h"

// Autograd backend
#include "autograd/pad/pad.h"

// Autocast backend
#include "autocast/pad/pad.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Padding operations
  m.def("pad(Tensor input, int[] padding, str mode, float value, int[]? dim, int order, Tensor? out) -> Tensor");
  m.def("pad_backward(Tensor grad_output, int[] input_shape, int[] padding, str mode, int[]? dim, int order) -> Tensor");
  m.def("pad_backward_backward(Tensor grad_grad_input, int[] padding, str mode, int[]? dim, int order) -> Tensor");
}
