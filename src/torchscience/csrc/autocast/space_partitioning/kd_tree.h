// src/torchscience/csrc/autocast/space_partitioning/kd_tree.h
#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::autocast::space_partitioning {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
kd_tree_build_batched(
    const at::Tensor& points,
    int64_t leaf_size
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Standard AMP behavior: cast float32/float64 inputs to float16
    // Users who need full precision can disable autocast for this call
    at::Tensor points_cast = at::autocast::cached_cast(at::kHalf, points);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::kd_tree_build_batched", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, int64_t
        )>()
        .call(points_cast, leaf_size);
}

}  // namespace torchscience::autocast::space_partitioning

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("kd_tree_build_batched", torchscience::autocast::space_partitioning::kd_tree_build_batched);
}
