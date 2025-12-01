#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor jacobi_polynomial_p_mulx(
    const at::Tensor& coeffs,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto exec_type = at::autocast::get_autocast_gpu_dtype();
    auto coeffs_casted = at::autocast::cached_cast(exec_type, coeffs);
    auto alpha_casted = at::autocast::cached_cast(exec_type, alpha);
    auto beta_casted = at::autocast::cached_cast(exec_type, beta);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::jacobi_polynomial_p_mulx", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(coeffs_casted, alpha_casted, beta_casted);
}

} // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("jacobi_polynomial_p_mulx", &torchscience::autocast::polynomial::jacobi_polynomial_p_mulx);
}
