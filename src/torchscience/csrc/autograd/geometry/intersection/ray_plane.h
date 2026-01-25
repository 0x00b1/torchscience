// src/torchscience/csrc/autograd/geometry/intersection/ray_plane.h
#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::geometry::intersection {

/**
 * Autograd function for ray-plane intersection.
 *
 * Computes the intersection of rays with infinite planes.
 */
class RayPlaneFunction : public torch::autograd::Function<RayPlaneFunction> {
 public:
  static std::vector<at::Tensor> forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& origins,
      const at::Tensor& directions,
      const at::Tensor& plane_normals,
      const at::Tensor& plane_offsets) {
    ctx->save_for_backward({origins, directions, plane_normals, plane_offsets});

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
                      .findSchemaOrThrow("torchscience::ray_plane", "")
                      .typed<std::tuple<
                          at::Tensor,
                          at::Tensor,
                          at::Tensor,
                          at::Tensor,
                          at::Tensor>(
                          const at::Tensor&,
                          const at::Tensor&,
                          const at::Tensor&,
                          const at::Tensor&)>()
                      .call(origins, directions, plane_normals, plane_offsets);

    // Save t and hit for backward
    ctx->saved_data["t"] = std::get<0>(result);
    ctx->saved_data["hit"] = std::get<4>(result);

    return {
        std::get<0>(result), // t
        std::get<1>(result), // hit_point
        std::get<2>(result), // normal
        std::get<3>(result), // uv
        std::get<4>(result)  // hit
    };
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor origins = saved[0];
    at::Tensor directions = saved[1];
    at::Tensor plane_normals = saved[2];
    at::Tensor plane_offsets = saved[3];

    at::Tensor t = ctx->saved_data["t"].toTensor();
    at::Tensor hit = ctx->saved_data["hit"].toTensor();

    at::Tensor grad_t = grad_outputs[0];
    at::Tensor grad_hit_point = grad_outputs[1];
    at::Tensor grad_normal = grad_outputs[2];
    at::Tensor grad_uv = grad_outputs[3];
    // grad_outputs[4] is grad_hit (boolean, no gradient)

    // Handle undefined gradients
    if (!grad_t.defined()) {
      grad_t = at::zeros_like(t);
    }
    if (!grad_hit_point.defined()) {
      // hit_point has shape (..., 3)
      std::vector<int64_t> point_shape = t.sizes().vec();
      point_shape.push_back(3);
      grad_hit_point = at::zeros(point_shape, origins.options());
    }
    if (!grad_normal.defined()) {
      // normal has shape (..., 3)
      std::vector<int64_t> normal_shape = t.sizes().vec();
      normal_shape.push_back(3);
      grad_normal = at::zeros(normal_shape, origins.options());
    }
    if (!grad_uv.defined()) {
      // uv has shape (..., 2)
      std::vector<int64_t> uv_shape = t.sizes().vec();
      uv_shape.push_back(2);
      grad_uv = at::zeros(uv_shape, origins.options());
    }

    at::AutoDispatchBelowAutograd guard;

    auto result =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("torchscience::ray_plane_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&,
                const at::Tensor&)>()
            .call(
                grad_t,
                grad_hit_point,
                grad_normal,
                grad_uv,
                origins,
                directions,
                plane_normals,
                plane_offsets,
                t,
                hit);

    return {
        std::get<0>(result), // grad_origins
        std::get<1>(result), // grad_directions
        std::get<2>(result), // grad_plane_normals
        std::get<3>(result)  // grad_plane_offsets
    };
  }
};

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ray_plane(
    const at::Tensor& origins,
    const at::Tensor& directions,
    const at::Tensor& plane_normals,
    const at::Tensor& plane_offsets) {
  auto result = RayPlaneFunction::apply(
      origins, directions, plane_normals, plane_offsets);
  return std::make_tuple(result[0], result[1], result[2], result[3], result[4]);
}

} // namespace torchscience::autograd::geometry::intersection

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl(
      "ray_plane",
      &torchscience::autograd::geometry::intersection::ray_plane);
}
