#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::graphics::shading {

namespace schlick_reflectance_impl {

/**
 * Reduce gradient to target size by summing over broadcasted dimensions.
 */
inline at::Tensor reduce_grad_to_size(const at::Tensor& grad, c10::IntArrayRef target_size) {
  if (grad.sizes() == target_size) {
    return grad;
  }

  // Handle empty target (scalar)
  if (target_size.empty()) {
    return grad.sum();
  }

  // Compute which dimensions to sum over
  std::vector<int64_t> dims_to_reduce;
  int64_t grad_dim = grad.dim();
  int64_t target_dim = static_cast<int64_t>(target_size.size());

  // Sum over leading dimensions that don't exist in target
  for (int64_t i = 0; i < grad_dim - target_dim; ++i) {
    dims_to_reduce.push_back(i);
  }

  // Sum over dimensions that were size 1 in target but expanded in grad
  for (int64_t i = 0; i < target_dim; ++i) {
    int64_t grad_idx = grad_dim - target_dim + i;
    if (target_size[i] == 1 && grad.size(grad_idx) != 1) {
      dims_to_reduce.push_back(grad_idx);
    }
  }

  if (dims_to_reduce.empty()) {
    return grad.view(target_size);
  }

  at::Tensor reduced = grad.sum(dims_to_reduce, /*keepdim=*/true);
  return reduced.view(target_size);
}

}  // namespace schlick_reflectance_impl

/**
 * Autograd function for Schlick reflectance.
 */
class SchlickReflectanceFunction : public torch::autograd::Function<SchlickReflectanceFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& cosine,
      const at::Tensor& r0
  ) {
    ctx->saved_data["cosine_requires_grad"] = cosine.requires_grad();
    ctx->saved_data["cosine_sizes"] = cosine.sizes().vec();

    ctx->save_for_backward({cosine, r0});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::schlick_reflectance", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(cosine, r0);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor cosine = saved[0];
    at::Tensor r0 = saved[1];

    at::Tensor grad_output = grad_outputs[0];

    bool cosine_requires_grad = ctx->saved_data["cosine_requires_grad"].toBool();
    auto cosine_sizes = ctx->saved_data["cosine_sizes"].toIntVector();

    if (!cosine_requires_grad || !grad_output.defined()) {
      return {at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    at::Tensor grad_cosine = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::schlick_reflectance_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, cosine, r0);

    // Reduce gradient to original cosine shape (handles broadcasting)
    grad_cosine = schlick_reflectance_impl::reduce_grad_to_size(grad_cosine, cosine_sizes);

    // No gradient for r0 (ior is not differentiable per spec)
    return {grad_cosine, at::Tensor()};
  }
};

inline at::Tensor schlick_reflectance(const at::Tensor& cosine, const at::Tensor& r0) {
  return SchlickReflectanceFunction::apply(cosine, r0);
}

}  // namespace torchscience::autograd::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("schlick_reflectance", &torchscience::autograd::graphics::shading::schlick_reflectance);
}
