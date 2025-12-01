#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor jacobi_polynomial_p_evaluate(
    const at::Tensor& coeffs,
    const at::Tensor& x,
    const at::Tensor& alpha,
    const at::Tensor& beta
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto device_type = coeffs.device().type();
    auto target_type = at::autocast::promote_type(at::kFloat, device_type, coeffs, x, alpha, beta);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::jacobi_polynomial_p_evaluate", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(
            at::autocast::cached_cast(target_type, coeffs),
            at::autocast::cached_cast(target_type, x),
            at::autocast::cached_cast(target_type, alpha),
            at::autocast::cached_cast(target_type, beta)
        );
}

}  // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("jacobi_polynomial_p_evaluate", torchscience::autocast::polynomial::jacobi_polynomial_p_evaluate);
}
