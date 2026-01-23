#pragma once

#include <vector>

#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

namespace torchscience::autocast::pad {

inline at::Tensor pad(
    const at::Tensor& input,
    std::vector<int64_t> padding,
    std::string mode,
    double value,
    c10::optional<std::vector<int64_t>> dim,
    int64_t order,
    c10::optional<at::Tensor> out
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // For padding, we generally want to preserve the input dtype
    // but cast to float for computation if half precision
    auto target_type = at::autocast::get_autocast_dtype(input.device().type());

    at::Tensor input_casted = at::autocast::cached_cast(target_type, input);

    c10::optional<at::Tensor> out_casted = c10::nullopt;
    if (out.has_value()) {
        out_casted = at::autocast::cached_cast(target_type, out.value());
    }

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::pad", "")
        .typed<at::Tensor(
            const at::Tensor&,
            std::vector<int64_t>,
            std::string,
            double,
            c10::optional<std::vector<int64_t>>,
            int64_t,
            c10::optional<at::Tensor>
        )>()
        .call(input_casted, padding, mode, value, dim, order, out_casted);
}

}  // namespace torchscience::autocast::pad

TORCH_LIBRARY_IMPL(torchscience, Autocast, module) {
    module.impl("pad", &torchscience::autocast::pad::pad);
}
