// src/torchscience/csrc/autocast/linear_algebra/symmetric_generalized_eigenvalue.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace torchscience::autocast::linear_algebra {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> symmetric_generalized_eigenvalue(
    const at::Tensor& a,
    const at::Tensor& b
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Promote to at least float32 for numerical stability
    auto target_dtype = at::kFloat;
    if (a.scalar_type() == at::kDouble || b.scalar_type() == at::kDouble) {
        target_dtype = at::kDouble;
    }

    auto a_casted = at::autocast::cached_cast(target_dtype, a);
    auto b_casted = at::autocast::cached_cast(target_dtype, b);

    return at::_ops::torchscience_symmetric_generalized_eigenvalue::call(a_casted, b_casted);
}

}  // namespace torchscience::autocast::linear_algebra

TORCH_LIBRARY_IMPL(torchscience, Autocast, m) {
    m.impl("symmetric_generalized_eigenvalue", &torchscience::autocast::linear_algebra::symmetric_generalized_eigenvalue);
}
