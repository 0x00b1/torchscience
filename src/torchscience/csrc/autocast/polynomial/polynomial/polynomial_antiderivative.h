#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor polynomial_antiderivative(
    const at::Tensor& coeffs,
    const at::Tensor& constant
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto device_type = coeffs.device().type();
    auto target_type = at::autocast::promote_type(at::kFloat, device_type, coeffs, constant);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::polynomial_antiderivative", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(
            at::autocast::cached_cast(target_type, coeffs),
            at::autocast::cached_cast(target_type, constant)
        );
}

}  // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("polynomial_antiderivative", torchscience::autocast::polynomial::polynomial_antiderivative);
}
