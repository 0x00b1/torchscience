#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

inline at::Tensor z_transform(
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor z_out_cast = at::autocast::cached_cast(target_dtype, z_out);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::z_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t)>()
        .call(input_cast, z_out_cast, dim);
}

inline at::Tensor z_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& z_out,
    int64_t dim
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor z_out_cast = at::autocast::cached_cast(target_dtype, z_out);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::z_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t)>()
        .call(grad_cast, input_cast, z_out_cast, dim);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("z_transform", &torchscience::autocast::transform::z_transform);
    module.impl("z_transform_backward", &torchscience::autocast::transform::z_transform_backward);
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl("z_transform", &torchscience::autocast::transform::z_transform);
    module.impl("z_transform_backward", &torchscience::autocast::transform::z_transform_backward);
}
