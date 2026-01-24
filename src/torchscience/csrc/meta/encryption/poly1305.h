#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// Poly1305 meta implementation (shape inference only)
at::Tensor poly1305(const at::Tensor& data, const at::Tensor& key) {
    TORCH_CHECK(data.dim() == 1,
        "poly1305: data must be a 1D tensor");
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "poly1305: key must be a 1D tensor of 32 bytes");
    return at::empty({16}, data.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("poly1305", &poly1305);
}

}  // namespace torchscience::meta::encryption
