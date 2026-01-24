#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// Ed25519 keypair generation meta implementation (shape inference only)
std::tuple<at::Tensor, at::Tensor> ed25519_keypair(const at::Tensor& seed) {
    TORCH_CHECK(seed.dim() == 1 && seed.size(0) == 32,
        "ed25519_keypair: seed must be a 1D tensor of 32 bytes");
    auto opts = seed.options();
    // Returns (private_key [64 bytes], public_key [32 bytes])
    return std::make_tuple(at::empty({64}, opts), at::empty({32}, opts));
}

// Ed25519 sign meta implementation (shape inference only)
at::Tensor ed25519_sign(const at::Tensor& private_key, const at::Tensor& message) {
    TORCH_CHECK(private_key.dim() == 1 && private_key.size(0) == 64,
        "ed25519_sign: private_key must be a 1D tensor of 64 bytes");
    TORCH_CHECK(message.dim() == 1,
        "ed25519_sign: message must be a 1D tensor");
    return at::empty({64}, private_key.options());
}

// Ed25519 verify meta implementation (shape inference only)
at::Tensor ed25519_verify(
    const at::Tensor& public_key,
    const at::Tensor& message,
    const at::Tensor& signature
) {
    TORCH_CHECK(public_key.dim() == 1 && public_key.size(0) == 32,
        "ed25519_verify: public_key must be a 1D tensor of 32 bytes");
    TORCH_CHECK(message.dim() == 1,
        "ed25519_verify: message must be a 1D tensor");
    TORCH_CHECK(signature.dim() == 1 && signature.size(0) == 64,
        "ed25519_verify: signature must be a 1D tensor of 64 bytes");
    return at::empty({}, at::TensorOptions().dtype(at::kByte).device(public_key.device()));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("ed25519_keypair", &ed25519_keypair);
    m.impl("ed25519_sign", &ed25519_sign);
    m.impl("ed25519_verify", &ed25519_verify);
}

}  // namespace torchscience::meta::encryption
