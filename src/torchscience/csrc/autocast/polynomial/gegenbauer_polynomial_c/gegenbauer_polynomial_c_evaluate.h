#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor gegenbauer_polynomial_c_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto device_type = coeffs.device().type();
    auto target_type = at::autocast::promote_type(at::kFloat, device_type, coeffs, x, alpha);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::gegenbauer_polynomial_c_evaluate", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(
            at::autocast::cached_cast(target_type, coeffs),
            at::autocast::cached_cast(target_type, x),
            at::autocast::cached_cast(target_type, alpha)
        );
}

}  // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("gegenbauer_polynomial_c_evaluate", torchscience::autocast::polynomial::gegenbauer_polynomial_c_evaluate);
}
