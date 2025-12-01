#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor chebyshev_polynomial_v_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto device_type = coeffs.device().type();
    auto target_type = at::autocast::promote_type(at::kFloat, device_type, coeffs, x);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chebyshev_polynomial_v_evaluate", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(
            at::autocast::cached_cast(target_type, coeffs),
            at::autocast::cached_cast(target_type, x)
        );
}

}  // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("chebyshev_polynomial_v_evaluate", torchscience::autocast::polynomial::chebyshev_polynomial_v_evaluate);
}
