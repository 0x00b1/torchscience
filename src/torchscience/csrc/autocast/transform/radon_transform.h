#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

inline at::Tensor radon_transform(
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor angles_cast = at::autocast::cached_cast(target_dtype, angles);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::radon_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, bool)>()
        .call(input_cast, angles_cast, circle);
}

inline at::Tensor radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& angles,
    bool circle
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor angles_cast = at::autocast::cached_cast(target_dtype, angles);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::radon_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, bool)>()
        .call(grad_cast, input_cast, angles_cast, circle);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("radon_transform", &torchscience::autocast::transform::radon_transform);
    module.impl("radon_transform_backward", &torchscience::autocast::transform::radon_transform_backward);
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl("radon_transform", &torchscience::autocast::transform::radon_transform);
    module.impl("radon_transform_backward", &torchscience::autocast::transform::radon_transform_backward);
}
