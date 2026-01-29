#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

/**
 * Autocast wrapper for discrete_wavelet_transform.
 */
inline at::Tensor discrete_wavelet_transform(
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor filter_lo_cast = at::autocast::cached_cast(target_dtype, filter_lo);
    at::Tensor filter_hi_cast = at::autocast::cached_cast(target_dtype, filter_hi);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::discrete_wavelet_transform", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(input_cast, filter_lo_cast, filter_hi_cast, levels, mode);
}

inline at::Tensor discrete_wavelet_transform_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& filter_lo,
    const at::Tensor& filter_hi,
    int64_t levels,
    int64_t mode,
    int64_t input_length
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor filter_lo_cast = at::autocast::cached_cast(target_dtype, filter_lo);
    at::Tensor filter_hi_cast = at::autocast::cached_cast(target_dtype, filter_hi);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::discrete_wavelet_transform_backward", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t, int64_t)>()
        .call(grad_cast, input_cast, filter_lo_cast, filter_hi_cast, levels, mode, input_length);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl(
        "discrete_wavelet_transform",
        &torchscience::autocast::transform::discrete_wavelet_transform
    );
    module.impl(
        "discrete_wavelet_transform_backward",
        &torchscience::autocast::transform::discrete_wavelet_transform_backward
    );
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl(
        "discrete_wavelet_transform",
        &torchscience::autocast::transform::discrete_wavelet_transform
    );
    module.impl(
        "discrete_wavelet_transform_backward",
        &torchscience::autocast::transform::discrete_wavelet_transform_backward
    );
}
