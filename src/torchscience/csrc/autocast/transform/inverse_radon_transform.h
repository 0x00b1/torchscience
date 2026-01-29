#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

inline at::Tensor inverse_radon_transform(
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        sinogram.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor sinogram_cast = at::autocast::cached_cast(target_dtype, sinogram);
    at::Tensor angles_cast = at::autocast::cached_cast(target_dtype, angles);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_radon_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, bool, int64_t, int64_t)>()
        .call(sinogram_cast, angles_cast, circle, output_size, filter_type);
}

inline at::Tensor inverse_radon_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& sinogram,
    const at::Tensor& angles,
    bool circle,
    int64_t output_size,
    int64_t filter_type
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor sinogram_cast = at::autocast::cached_cast(target_dtype, sinogram);
    at::Tensor angles_cast = at::autocast::cached_cast(target_dtype, angles);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_radon_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, bool, int64_t, int64_t)>()
        .call(grad_cast, sinogram_cast, angles_cast, circle, output_size, filter_type);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("inverse_radon_transform", &torchscience::autocast::transform::inverse_radon_transform);
    module.impl("inverse_radon_transform_backward", &torchscience::autocast::transform::inverse_radon_transform_backward);
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl("inverse_radon_transform", &torchscience::autocast::transform::inverse_radon_transform);
    module.impl("inverse_radon_transform_backward", &torchscience::autocast::transform::inverse_radon_transform_backward);
}
