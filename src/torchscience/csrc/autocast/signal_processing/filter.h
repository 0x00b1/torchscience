#pragma once

#include <torch/library.h>
#include <ATen/autocast_mode.h>

namespace torchscience::autocast::filter {

/**
 * Autocast wrapper for butterworth_analog_bandpass_filter.
 * Handles automatic mixed precision casting.
 */
inline at::Tensor butterworth_analog_bandpass_filter(
    int64_t n,
    const at::Tensor& omega_p1,
    const at::Tensor& omega_p2
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Determine target dtype based on autocast settings
    auto target_dtype = at::autocast::get_autocast_dtype(
        omega_p1.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    // Cast inputs to target dtype
    at::Tensor omega_p1_cast = at::autocast::cached_cast(target_dtype, omega_p1);
    at::Tensor omega_p2_cast = at::autocast::cached_cast(target_dtype, omega_p2);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::butterworth_analog_bandpass_filter", "")
        .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&)>()
        .call(n, omega_p1_cast, omega_p2_cast);
}

}  // namespace torchscience::autocast::filter

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    module.impl(
        "butterworth_analog_bandpass_filter",
        &torchscience::autocast::filter::butterworth_analog_bandpass_filter
    );
}
