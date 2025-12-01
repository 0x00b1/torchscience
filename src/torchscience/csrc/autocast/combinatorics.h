#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::combinatorics {

inline at::Tensor binomial_coefficient(
  const at::Tensor &n_input,
  const at::Tensor &k_input
) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

  auto target_dtype = at::promote_types(n_input.scalar_type(), k_input.scalar_type());

  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::binomial_coefficient", "")
    .typed<at::Tensor(const at::Tensor &, const at::Tensor &)>()
    .call(
      at::autocast::cached_cast(target_dtype, n_input, c10::DeviceType::CUDA),
      at::autocast::cached_cast(target_dtype, k_input, c10::DeviceType::CUDA)
    );
}

} // namespace torchscience::autocast::combinatorics

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
  module.impl(
    "binomial_coefficient",
    torchscience::autocast::combinatorics::binomial_coefficient
  );
}
