#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor chebyshev_polynomial_w_antiderivative(
    const at::Tensor& coeffs
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto exec_type = at::autocast::get_autocast_gpu_dtype();
    auto coeffs_casted = at::autocast::cached_cast(exec_type, coeffs);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chebyshev_polynomial_w_antiderivative", "")
        .typed<at::Tensor(const at::Tensor&)>()
        .call(coeffs_casted);
}

} // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("chebyshev_polynomial_w_antiderivative", &torchscience::autocast::polynomial::chebyshev_polynomial_w_antiderivative);
}
