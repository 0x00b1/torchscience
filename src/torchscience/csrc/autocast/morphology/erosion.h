#pragma once

#include <torch/library.h>
#include <ATen/autocast_mode.h>

namespace torchscience::autocast::morphology {

/**
 * Autocast wrapper for erosion.
 * Handles automatic mixed precision casting.
 */
inline at::Tensor erosion(
    const at::Tensor& input,
    const at::Tensor& structuring_element,
    c10::optional<at::IntArrayRef> origin,
    int64_t padding_mode
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Determine target dtype based on autocast settings
    auto target_dtype = at::autocast::get_autocast_dtype(
        input.device().is_cpu() ? at::kCPU : at::kCUDA
    );

    // Cast input to target dtype
    at::Tensor input_cast = at::autocast::cached_cast(target_dtype, input);

    // SE may be bool (flat) or float (non-flat)
    at::Tensor se_cast = structuring_element;
    if (structuring_element.dtype() != at::kBool) {
        se_cast = at::autocast::cached_cast(target_dtype, structuring_element);
    }

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::erosion", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, c10::optional<at::IntArrayRef>, int64_t)>()
        .call(input_cast, se_cast, origin, padding_mode);
}

}  // namespace torchscience::autocast::morphology

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("erosion", &torchscience::autocast::morphology::erosion);
}

TORCH_LIBRARY_IMPL(torchscience, AutocastCUDA, module) {
    module.impl("erosion", &torchscience::autocast::morphology::erosion);
}
