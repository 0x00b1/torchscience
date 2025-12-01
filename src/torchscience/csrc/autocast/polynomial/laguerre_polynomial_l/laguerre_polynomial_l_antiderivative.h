#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor laguerre_polynomial_l_antiderivative(const at::Tensor& coeffs) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto exec_type = at::autocast::get_autocast_gpu_dtype();
    auto coeffs_casted = at::autocast::cached_cast(exec_type, coeffs);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::laguerre_polynomial_l_antiderivative", "")
        .typed<at::Tensor(const at::Tensor&)>()
        .call(coeffs_casted);
}

} // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("laguerre_polynomial_l_antiderivative", &torchscience::autocast::polynomial::laguerre_polynomial_l_antiderivative);
}
