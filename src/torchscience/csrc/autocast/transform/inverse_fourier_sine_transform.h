#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

/**
 * Autocast wrapper for inverse_fourier_sine_transform (IDST).
 */
inline at::Tensor inverse_fourier_sine_transform(
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_fourier_sine_transform", "")
        .typed<at::Tensor(const at::Tensor&, int64_t, int64_t, int64_t, int64_t)>()
        .call(input_cast, n_param, dim, type, norm);
}

inline at::Tensor inverse_fourier_sine_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    int64_t n_param,
    int64_t dim,
    int64_t type,
    int64_t norm
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_fourier_sine_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t, int64_t)>()
        .call(grad_cast, input_cast, n_param, dim, type, norm);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl(
        "inverse_fourier_sine_transform",
        &torchscience::autocast::transform::inverse_fourier_sine_transform
    );
    module.impl(
        "inverse_fourier_sine_transform_backward",
        &torchscience::autocast::transform::inverse_fourier_sine_transform_backward
    );
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl(
        "inverse_fourier_sine_transform",
        &torchscience::autocast::transform::inverse_fourier_sine_transform
    );
    module.impl(
        "inverse_fourier_sine_transform_backward",
        &torchscience::autocast::transform::inverse_fourier_sine_transform_backward
    );
}
