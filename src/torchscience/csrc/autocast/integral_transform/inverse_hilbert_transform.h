#pragma once

#include <torch/library.h>
#include <ATen/autocast_mode.h>

namespace torchscience::autocast::integral_transform {

/**
 * Autocast wrapper for inverse_hilbert_transform.
 * Handles automatic mixed precision casting.
 */
inline at::Tensor inverse_hilbert_transform(
    const at::Tensor& input,
    int64_t n,
    int64_t dim,
    int64_t padding_mode,
    double padding_value,
    const c10::optional<at::Tensor>& window
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Determine target dtype based on autocast settings
    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    // Cast input to target dtype
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);

    // Cast window if provided
    c10::optional<at::Tensor> window_cast = c10::nullopt;
    if (window.has_value()) {
        window_cast = at::autocast::cached_cast(target_dtype, window.value());
    }

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::inverse_hilbert_transform", "")
        .typed<at::Tensor(const at::Tensor&, int64_t, int64_t, int64_t, double, const c10::optional<at::Tensor>&)>()
        .call(input_cast, n, dim, padding_mode, padding_value, window_cast);
}

}  // namespace torchscience::autocast::integral_transform

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::autocast::integral_transform::inverse_hilbert_transform
    );
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl(
        "inverse_hilbert_transform",
        &torchscience::autocast::integral_transform::inverse_hilbert_transform
    );
}
