#pragma once

#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../kernel/pad/normalize.h"

namespace torchscience::meta::pad {

using namespace torchscience::kernel::pad;

inline at::Tensor pad(
    const at::Tensor& input,
    const std::vector<int64_t>& padding,
    const std::string& mode_str,
    double value,
    const c10::optional<std::vector<int64_t>>& dim,
    int64_t order,
    const c10::optional<at::Tensor>& out
) {
    int64_t ndim = input.dim();
    auto [dims, amounts] = normalize_padding(padding, dim, ndim);
    std::vector<int64_t> output_shape = compute_output_shape(
        input.sizes().vec(), dims, amounts
    );

    if (out.has_value()) {
        return out.value();
    }
    return at::empty(output_shape, input.options());
}

inline at::Tensor pad_backward(
    const at::Tensor& grad_output,
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& padding,
    const std::string& mode_str,
    const c10::optional<std::vector<int64_t>>& dim,
    int64_t order
) {
    return at::empty(input_shape, grad_output.options());
}

inline at::Tensor pad_backward_backward(
    const at::Tensor& grad_grad_input,
    const std::vector<int64_t>& padding,
    const std::string& mode_str,
    const c10::optional<std::vector<int64_t>>& dim,
    int64_t order
) {
    int64_t ndim = grad_grad_input.dim();
    auto [dims, amounts] = normalize_padding(padding, dim, ndim);
    std::vector<int64_t> output_shape = compute_output_shape(
        grad_grad_input.sizes().vec(), dims, amounts
    );
    return at::empty(output_shape, grad_grad_input.options());
}

}  // namespace torchscience::meta::pad

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl("pad", &torchscience::meta::pad::pad);
    module.impl("pad_backward", &torchscience::meta::pad::pad_backward);
    module.impl("pad_backward_backward", &torchscience::meta::pad::pad_backward_backward);
}
