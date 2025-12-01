#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace torchscience::meta::signal_processing::noise {

inline at::Tensor white_noise(
  at::IntArrayRef size,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad,
  const c10::optional<at::Generator> generator
) {
  TORCH_CHECK(size.size() > 0, "white_noise: size must be non-empty");
  for (auto s : size) {
    TORCH_CHECK(s >= 0, "white_noise: size elements must be non-negative, got ", s);
  }

  // Determine output dtype
  at::ScalarType out_dtype = dtype.value_or(
    c10::typeMetaToScalarType(at::get_default_dtype())
  );
  at::Layout out_layout = layout.value_or(at::kStrided);

  // Meta tensors always use Meta device
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(out_layout)
    .device(at::kMeta)
    .requires_grad(requires_grad);

  return at::empty(size.vec(), options);
}

} // namespace torchscience::meta::signal_processing::noise

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
  module.impl("white_noise", &torchscience::meta::signal_processing::noise::white_noise);
}
