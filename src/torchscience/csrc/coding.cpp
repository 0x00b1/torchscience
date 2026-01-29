// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/coding/morton.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Morton encoding (Z-order curve)
  m.def("morton_encode(Tensor coordinates) -> Tensor");
  m.def("morton_decode(Tensor codes, int dimensions) -> Tensor");
}
