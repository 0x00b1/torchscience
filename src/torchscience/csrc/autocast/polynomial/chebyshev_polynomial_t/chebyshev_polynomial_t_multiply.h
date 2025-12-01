#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor chebyshev_polynomial_t_multiply(
    const at::Tensor& a,
    const at::Tensor& b
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto exec_type = at::autocast::get_autocast_gpu_dtype();
    auto a_casted = at::autocast::cached_cast(exec_type, a);
    auto b_casted = at::autocast::cached_cast(exec_type, b);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::chebyshev_polynomial_t_multiply", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(a_casted, b_casted);
}

} // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("chebyshev_polynomial_t_multiply", &torchscience::autocast::polynomial::chebyshev_polynomial_t_multiply);
}
