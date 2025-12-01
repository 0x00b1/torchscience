#pragma once

#include <tuple>

#include <torch/library.h>

namespace torchscience::meta::descriptive {

inline std::tuple<at::Tensor, at::Tensor> histogram(
    const at::Tensor& input,
    int64_t bins,
    [[maybe_unused]] c10::optional<at::ArrayRef<double>> range,
    [[maybe_unused]] const c10::optional<at::Tensor>& weights,
    [[maybe_unused]] bool density,
    [[maybe_unused]] c10::string_view closed,
    [[maybe_unused]] c10::string_view out_of_bounds
) {
    auto options = input.options();
    at::Tensor counts = at::empty({bins}, options);
    at::Tensor edges = at::empty({bins + 1}, options);
    return std::make_tuple(counts, edges);
}

inline std::tuple<at::Tensor, at::Tensor> histogram_edges(
    const at::Tensor& input,
    const at::Tensor& edges,
    [[maybe_unused]] const c10::optional<at::Tensor>& weights,
    [[maybe_unused]] bool density,
    [[maybe_unused]] c10::string_view closed,
    [[maybe_unused]] c10::string_view out_of_bounds
) {
    int64_t bins = edges.numel() - 1;
    auto options = input.options();
    at::Tensor counts = at::empty({bins}, options);
    return std::make_tuple(counts, edges.clone());
}

}  // namespace torchscience::meta::descriptive

TORCH_LIBRARY_IMPL(torchscience, Meta, module) {
    module.impl(
        "histogram",
        &torchscience::meta::descriptive::histogram
    );

    module.impl(
        "histogram_edges",
        &torchscience::meta::descriptive::histogram_edges
    );
}
