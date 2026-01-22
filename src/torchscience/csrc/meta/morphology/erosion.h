#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::morphology {

/**
 * Meta implementation for erosion shape inference.
 */
inline at::Tensor erosion(
    const at::Tensor& input,
    const at::Tensor& structuring_element,
    c10::optional<at::IntArrayRef> origin,
    int64_t padding_mode
) {
    TORCH_CHECK(input.dim() >= structuring_element.dim(),
        "erosion: input must have at least as many dimensions as structuring_element");
    // Output shape is same as input shape
    return at::empty_like(input);
}

/**
 * Meta implementation for erosion backward shape inference.
 */
inline at::Tensor erosion_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& structuring_element,
    c10::optional<at::IntArrayRef> origin,
    int64_t padding_mode
) {
    return at::empty_like(input);
}

}  // namespace torchscience::meta::morphology

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("erosion", &torchscience::meta::morphology::erosion);
    m.impl("erosion_backward", &torchscience::meta::morphology::erosion_backward);
}
