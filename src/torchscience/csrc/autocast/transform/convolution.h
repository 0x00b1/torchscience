#pragma once

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::transform {

/**
 * Autocast wrapper for convolution.
 */
inline at::Tensor convolution(
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor kernel_cast = at::autocast::cached_cast(target_dtype, kernel);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::convolution", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(input_cast, kernel_cast, dim, mode);
}

inline std::tuple<at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& kernel,
    int64_t dim,
    int64_t mode
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    auto target_dtype = at::autocast::get_autocast_dtype(
        grad_output.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    at::Tensor grad_cast = at::autocast::cached_cast(target_dtype, grad_output);
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);
    at::Tensor kernel_cast = at::autocast::cached_cast(target_dtype, kernel);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::convolution_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t)>()
        .call(grad_cast, input_cast, kernel_cast, dim, mode);
}

}  // namespace torchscience::autocast::transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl(
        "convolution",
        &torchscience::autocast::transform::convolution
    );
    module.impl(
        "convolution_backward",
        &torchscience::autocast::transform::convolution_backward
    );
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl(
        "convolution",
        &torchscience::autocast::transform::convolution
    );
    module.impl(
        "convolution_backward",
        &torchscience::autocast::transform::convolution_backward
    );
}
