#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

at::Tensor sha3_256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "sha3_256: data must be uint8");
    return at::empty({32}, data.options());
}

at::Tensor sha3_512(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "sha3_512: data must be uint8");
    return at::empty({64}, data.options());
}

at::Tensor keccak256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "keccak256: data must be uint8");
    return at::empty({32}, data.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("sha3_256", &sha3_256);
    m.impl("sha3_512", &sha3_512);
    m.impl("keccak256", &keccak256);
}

}  // namespace torchscience::meta::encryption
