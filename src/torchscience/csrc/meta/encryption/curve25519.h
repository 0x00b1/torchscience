#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// X25519 scalar multiplication meta implementation (shape inference only)
at::Tensor x25519(const at::Tensor& scalar, const at::Tensor& point) {
    TORCH_CHECK(scalar.dim() == 1 && scalar.size(0) == 32,
        "x25519: scalar must be a 1D tensor of 32 bytes");
    TORCH_CHECK(point.dim() == 1 && point.size(0) == 32,
        "x25519: point must be a 1D tensor of 32 bytes");
    return at::empty({32}, scalar.options());
}

// X25519 base point multiplication meta implementation (shape inference only)
at::Tensor x25519_base(const at::Tensor& scalar) {
    TORCH_CHECK(scalar.dim() == 1 && scalar.size(0) == 32,
        "x25519_base: scalar must be a 1D tensor of 32 bytes");
    return at::empty({32}, scalar.options());
}

// X25519 keypair generation meta implementation (shape inference only)
std::tuple<at::Tensor, at::Tensor> x25519_keypair(const at::Tensor& seed) {
    TORCH_CHECK(seed.dim() == 1 && seed.size(0) == 32,
        "x25519_keypair: seed must be a 1D tensor of 32 bytes");
    auto opts = seed.options();
    return std::make_tuple(at::empty({32}, opts), at::empty({32}, opts));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("x25519", &x25519);
    m.impl("x25519_base", &x25519_base);
    m.impl("x25519_keypair", &x25519_keypair);
}

}  // namespace torchscience::meta::encryption
