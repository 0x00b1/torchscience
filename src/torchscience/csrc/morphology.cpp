// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/morphology/erosion.h"
#include "cpu/morphology/dilation.h"

// Meta backend
#include "meta/morphology/erosion.h"
#include "meta/morphology/dilation.h"

// Autograd backend
#include "autograd/morphology/erosion.h"
#include "autograd/morphology/dilation.h"

// Autocast backend
#include "autocast/morphology/erosion.h"
#include "autocast/morphology/dilation.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Morphological erosion
  m.def("erosion(Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");
  m.def("erosion_backward(Tensor grad_output, Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");

  // Morphological dilation
  m.def("dilation(Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");
  m.def("dilation_backward(Tensor grad_output, Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");
}
