// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Gradient operator - computes spatial gradient of scalar field
  // Input: field (*batch, *spatial), spacing (ndim,), dims (ndim,)
  // Output: gradient (ndim, *batch, *spatial)
  m.def("gradient_nd(Tensor field, Tensor spacing, int[] dims, str boundary) -> Tensor");
  m.def("gradient_nd_backward(Tensor grad_output, Tensor field, Tensor spacing, int[] dims, str boundary) -> Tensor");

  // Laplacian operator - computes Laplacian of scalar field
  // Input: field (*batch, *spatial), spacing (ndim,), dims (ndim,)
  // Output: laplacian (*batch, *spatial)
  m.def("laplacian_nd(Tensor field, Tensor spacing, int[] dims, str boundary) -> Tensor");
  m.def("laplacian_nd_backward(Tensor grad_output, Tensor field, Tensor spacing, int[] dims, str boundary) -> Tensor");

  // Divergence operator - computes divergence of vector field
  // Input: vector_field (ndim, *batch, *spatial), spacing (ndim,), dims (ndim,)
  // Output: scalar field (*batch, *spatial)
  m.def("divergence_nd(Tensor vector_field, Tensor spacing, int[] dims, str boundary) -> Tensor");
  m.def("divergence_nd_backward(Tensor grad_output, Tensor vector_field, Tensor spacing, int[] dims, str boundary) -> Tensor");

  // Curl operator - 3D only
  // Input: vector_field (3, *batch, H, W, D), spacing (3,), dims (3,)
  // Output: curl (3, *batch, H, W, D)
  m.def("curl_3d(Tensor vector_field, Tensor spacing, int[] dims, str boundary) -> Tensor");
  m.def("curl_3d_backward(Tensor grad_output, Tensor vector_field, Tensor spacing, int[] dims, str boundary) -> Tensor");
}
