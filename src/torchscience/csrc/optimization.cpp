// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// Test functions
#include "composite/optimization/test_functions.h"
#include "cpu/optimization/test_functions.h"
#include "meta/optimization/test_functions.h"
#include "autograd/optimization/test_functions.h"

// Combinatorial optimization
#include "cpu/optimization/combinatorial.h"
#include "meta/optimization/combinatorial.h"
#include "autograd/optimization/combinatorial.h"

// Sparse backends
#include "sparse/coo/cpu/optimization/test_functions.h"
#include "sparse/csr/cpu/optimization/test_functions.h"

// Quantized backends
#include "quantized/cpu/optimization/test_functions.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/optimization/test_functions.cu"
#include "sparse/coo/cuda/optimization/test_functions.h"
#include "sparse/csr/cuda/optimization/test_functions.h"
#include "quantized/cuda/optimization/test_functions.h"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Test functions for optimization
  m.def("rosenbrock(Tensor x, Tensor a, Tensor b) -> Tensor");
  m.def("rosenbrock_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("rosenbrock_backward_backward(Tensor gg_x, Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  m.def("sphere(Tensor x) -> Tensor");
  m.def("booth(Tensor x1, Tensor x2) -> Tensor");
  m.def("beale(Tensor x1, Tensor x2) -> Tensor");
  m.def("himmelblau(Tensor x1, Tensor x2) -> Tensor");
  m.def("rastrigin(Tensor x) -> Tensor");
  m.def("ackley(Tensor x) -> Tensor");

  // Combinatorial optimization
  m.def("sinkhorn(Tensor C, Tensor a, Tensor b, float epsilon, int maxiter, float tol) -> Tensor");
  m.def("sinkhorn_backward(Tensor grad_output, Tensor P, Tensor C, float epsilon) -> Tensor");
}
