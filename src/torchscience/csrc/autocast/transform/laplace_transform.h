#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

inline at::Tensor laplace_transform(
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor s_cast = at::autocast::cached_cast(target_dtype, s);
    at::Tensor t_cast = at::autocast::cached_cast(target_dtype, t);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::laplace_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(input_cast, s_cast, t_cast, dim, integration_method);
}

inline at::Tensor laplace_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& s,
    const at::Tensor& t,
    int64_t dim,
    int64_t integration_method
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor s_cast = at::autocast::cached_cast(target_dtype, s);
    at::Tensor t_cast = at::autocast::cached_cast(target_dtype, t);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::laplace_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(grad_cast, input_cast, s_cast, t_cast, dim, integration_method);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("laplace_transform", &torchscience::autocast::transform::laplace_transform);
    module.impl("laplace_transform_backward", &torchscience::autocast::transform::laplace_transform_backward);
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl("laplace_transform", &torchscience::autocast::transform::laplace_transform);
    module.impl("laplace_transform_backward", &torchscience::autocast::transform::laplace_transform_backward);
}
