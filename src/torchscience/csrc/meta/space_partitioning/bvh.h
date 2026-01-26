// src/torchscience/csrc/meta/space_partitioning/bvh.h
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::space_partitioning {

/**
 * Meta implementation of BVH build for shape inference.
 *
 * @param vertices Mesh vertices, shape (V, 3)
 * @param faces Triangle indices, shape (F, 3)
 * @return Tensor of shape [1] containing scene handle (int64)
 */
inline at::Tensor bvh_build(
    const at::Tensor& vertices,
    const at::Tensor& faces) {
  TORCH_CHECK(
      vertices.dim() == 2 && vertices.size(1) == 3,
      "bvh_build: vertices must be (V, 3), got ",
      vertices.sizes());
  TORCH_CHECK(
      faces.dim() == 2 && faces.size(1) == 3,
      "bvh_build: faces must be (F, 3), got ",
      faces.sizes());

  return at::empty({1}, vertices.options().dtype(at::kLong));
}

/**
 * Meta implementation of BVH destroy (no-op for shape inference).
 */
inline void bvh_destroy(int64_t scene_handle) {
  // No-op for meta tensors
  (void)scene_handle;
}

}  // namespace torchscience::meta::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("bvh_build", torchscience::meta::space_partitioning::bvh_build);
  m.impl("bvh_destroy", torchscience::meta::space_partitioning::bvh_destroy);
}
