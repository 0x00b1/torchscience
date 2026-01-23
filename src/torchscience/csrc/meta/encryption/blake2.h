#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

at::Tensor blake2b(const at::Tensor& data, const at::Tensor& key, int64_t digest_size) {
    TORCH_CHECK(data.dtype() == at::kByte, "blake2b: data must be uint8");
    TORCH_CHECK(key.dtype() == at::kByte, "blake2b: key must be uint8");
    TORCH_CHECK(digest_size >= 1 && digest_size <= 64, "blake2b: digest_size must be 1-64");
    TORCH_CHECK(key.numel() <= 64, "blake2b: key must be <= 64 bytes");
    return at::empty({digest_size}, data.options());
}

at::Tensor blake2s(const at::Tensor& data, const at::Tensor& key, int64_t digest_size) {
    TORCH_CHECK(data.dtype() == at::kByte, "blake2s: data must be uint8");
    TORCH_CHECK(key.dtype() == at::kByte, "blake2s: key must be uint8");
    TORCH_CHECK(digest_size >= 1 && digest_size <= 32, "blake2s: digest_size must be 1-32");
    TORCH_CHECK(key.numel() <= 32, "blake2s: key must be <= 32 bytes");
    return at::empty({digest_size}, data.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("blake2b", &blake2b);
    m.impl("blake2s", &blake2s);
}

}  // namespace torchscience::meta::encryption
