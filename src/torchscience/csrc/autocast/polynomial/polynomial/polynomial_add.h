#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::polynomial {

inline at::Tensor polynomial_add(
    const at::Tensor& p,
    const at::Tensor& q
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto device_type = p.device().type();
    auto target_type = at::autocast::promote_type(at::kFloat, device_type, p, q);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::polynomial_add", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&)>()
        .call(
            at::autocast::cached_cast(target_type, p),
            at::autocast::cached_cast(target_type, q)
        );
}

}  // namespace torchscience::autocast::polynomial

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("polynomial_add", torchscience::autocast::polynomial::polynomial_add);
}
