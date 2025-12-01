#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::tone_mapping {

inline at::Tensor reinhard(
    const at::Tensor& input,
    const std::optional<at::Tensor>& white_point
) {
    return at::empty_like(input);
}

inline std::tuple<at::Tensor, at::Tensor> reinhard_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const std::optional<at::Tensor>& white_point
) {
    return std::make_tuple(
        at::empty_like(input),
        at::empty({}, input.options())
    );
}

}  // namespace torchscience::meta::graphics::tone_mapping

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("reinhard", &torchscience::meta::graphics::tone_mapping::reinhard);
    m.impl("reinhard_backward", &torchscience::meta::graphics::tone_mapping::reinhard_backward);
}
