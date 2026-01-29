// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/privacy/gaussian_mechanism.h"
#include "cpu/privacy/laplace_mechanism.h"

// Meta backend
#include "meta/privacy/gaussian_mechanism.h"
#include "meta/privacy/laplace_mechanism.h"

// Autograd backend
#include "autograd/privacy/gaussian_mechanism.h"
#include "autograd/privacy/laplace_mechanism.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Differential privacy mechanisms
  m.def("gaussian_mechanism(Tensor x, Tensor noise, float sigma) -> Tensor");
  m.def("gaussian_mechanism_backward(Tensor grad_output) -> Tensor");

  m.def("laplace_mechanism(Tensor x, Tensor noise, float b) -> Tensor");
  m.def("laplace_mechanism_backward(Tensor grad_output) -> Tensor");
}
