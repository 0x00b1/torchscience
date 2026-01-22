#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::morphology {

/**
 * Meta implementation for dilation shape inference.
 */
inline at::Tensor dilation(
    const at::Tensor& input,
    const at::Tensor& structuring_element,
    c10::optional<at::IntArrayRef> origin,
    int64_t padding_mode
) {
    TORCH_CHECK(input.dim() >= structuring_element.dim(),
        "dilation: input must have at least as many dimensions as structuring_element");
    // Output shape is same as input shape
    return at::empty_like(input);
}

/**
 * Meta implementation for dilation backward shape inference.
 */
inline at::Tensor dilation_backward(
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
    m.impl("dilation", &torchscience::meta::morphology::dilation);
    m.impl("dilation_backward", &torchscience::meta::morphology::dilation_backward);
}
