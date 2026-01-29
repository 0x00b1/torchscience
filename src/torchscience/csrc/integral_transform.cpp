// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/integral_transform/hilbert_transform.h"
#include "cpu/integral_transform/inverse_hilbert_transform.h"

// Meta backend
#include "meta/integral_transform/hilbert_transform.h"
#include "meta/integral_transform/inverse_hilbert_transform.h"

// Autograd backend
#include "autograd/integral_transform/hilbert_transform.h"
#include "autograd/integral_transform/inverse_hilbert_transform.h"

// Autocast backend
#include "autocast/integral_transform/hilbert_transform.h"
#include "autocast/integral_transform/inverse_hilbert_transform.h"

// Sparse backends
#include "sparse/coo/cpu/integral_transform/hilbert_transform.h"
#include "sparse/coo/cpu/integral_transform/inverse_hilbert_transform.h"
#include "sparse/csr/cpu/integral_transform/hilbert_transform.h"
#include "sparse/csr/cpu/integral_transform/inverse_hilbert_transform.h"

// Quantized backends
#include "quantized/cpu/integral_transform/hilbert_transform.h"
#include "quantized/cpu/integral_transform/inverse_hilbert_transform.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/integral_transform/hilbert_transform.cu"
#include "cuda/integral_transform/inverse_hilbert_transform.cu"
#include "sparse/coo/cuda/integral_transform/hilbert_transform.h"
#include "sparse/coo/cuda/integral_transform/inverse_hilbert_transform.h"
#include "sparse/csr/cuda/integral_transform/hilbert_transform.h"
#include "sparse/csr/cuda/integral_transform/inverse_hilbert_transform.h"
#include "quantized/cuda/integral_transform/hilbert_transform.h"
#include "quantized/cuda/integral_transform/inverse_hilbert_transform.h"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Hilbert transform
  m.def("hilbert_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  m.def("hilbert_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  m.def("hilbert_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

  // Inverse Hilbert transform
  m.def("inverse_hilbert_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  m.def("inverse_hilbert_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  m.def("inverse_hilbert_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");
}
