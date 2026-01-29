#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

inline at::Tensor abel_transform(
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor y_out_cast = at::autocast::cached_cast(target_dtype, y_out);
    at::Tensor r_in_cast = at::autocast::cached_cast(target_dtype, r_in);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::abel_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(input_cast, y_out_cast, r_in_cast, dim, integration_method);
}

inline at::Tensor abel_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& y_out,
    const at::Tensor& r_in,
    int64_t dim,
    int64_t integration_method
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor y_out_cast = at::autocast::cached_cast(target_dtype, y_out);
    at::Tensor r_in_cast = at::autocast::cached_cast(target_dtype, r_in);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::abel_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(grad_cast, input_cast, y_out_cast, r_in_cast, dim, integration_method);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("abel_transform", &torchscience::autocast::transform::abel_transform);
    module.impl("abel_transform_backward", &torchscience::autocast::transform::abel_transform_backward);
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl("abel_transform", &torchscience::autocast::transform::abel_transform);
    module.impl("abel_transform_backward", &torchscience::autocast::transform::abel_transform_backward);
}
