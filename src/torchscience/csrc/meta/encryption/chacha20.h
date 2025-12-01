#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

at::Tensor chacha20(
    const at::Tensor& key,
    const at::Tensor& nonce,
    int64_t num_bytes,
    int64_t counter
) {
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20: key must be a 1D tensor of 32 bytes");
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20: nonce must be a 1D tensor of 12 bytes");
    TORCH_CHECK(num_bytes > 0,
        "chacha20: num_bytes must be positive");

    return at::empty({num_bytes}, key.options().dtype(at::kByte));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("chacha20", &chacha20);
}

}  // namespace torchscience::meta::encryption
