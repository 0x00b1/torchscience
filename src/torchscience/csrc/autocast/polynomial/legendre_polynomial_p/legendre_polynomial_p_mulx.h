#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor legendre_polynomial_p_mulx(const at::Tensor& coeffs) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto exec_type = at::autocast::get_autocast_gpu_dtype();
    auto coeffs_casted = at::autocast::cached_cast(exec_type, coeffs);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::legendre_polynomial_p_mulx", "")
        .typed<at::Tensor(const at::Tensor&)>()
        .call(coeffs_casted);
}

} // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("legendre_polynomial_p_mulx", &torchscience::autocast::polynomial::legendre_polynomial_p_mulx);
}
