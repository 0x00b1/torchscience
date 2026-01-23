#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/ed25519.h"

namespace torchscience::cpu::encryption {

// Generate Ed25519 keypair from seed
// seed: 1D tensor of 32 bytes
// Returns: tuple of (private_key [64 bytes], public_key [32 bytes])
std::tuple<at::Tensor, at::Tensor> ed25519_keypair(const at::Tensor& seed) {
    TORCH_CHECK(seed.dim() == 1 && seed.size(0) == 32,
        "ed25519_keypair: seed must be a 1D tensor of 32 bytes, got shape ", seed.sizes());
    TORCH_CHECK(seed.dtype() == at::kByte,
        "ed25519_keypair: seed must be uint8, got ", seed.dtype());

    auto seed_contig = seed.contiguous();

    auto public_key = at::empty({32}, seed.options());
    auto private_key = at::empty({64}, seed.options());

    kernel::encryption::ed25519_create_keypair(
        public_key.data_ptr<uint8_t>(),
        private_key.data_ptr<uint8_t>(),
        seed_contig.data_ptr<uint8_t>()
    );

    return std::make_tuple(private_key, public_key);
}

// Sign message with Ed25519
// private_key: 1D tensor of 64 bytes (from ed25519_keypair)
// message: 1D tensor of arbitrary length
// Returns: 1D tensor of 64 bytes (signature)
at::Tensor ed25519_sign(const at::Tensor& private_key, const at::Tensor& message) {
    TORCH_CHECK(private_key.dim() == 1 && private_key.size(0) == 64,
        "ed25519_sign: private_key must be a 1D tensor of 64 bytes, got shape ", private_key.sizes());
    TORCH_CHECK(private_key.dtype() == at::kByte,
        "ed25519_sign: private_key must be uint8, got ", private_key.dtype());
    TORCH_CHECK(message.dim() == 1,
        "ed25519_sign: message must be a 1D tensor, got ", message.dim(), " dimensions");
    TORCH_CHECK(message.dtype() == at::kByte,
        "ed25519_sign: message must be uint8, got ", message.dtype());

    auto private_key_contig = private_key.contiguous();
    auto message_contig = message.contiguous();

    auto signature = at::empty({64}, private_key.options());

    kernel::encryption::ed25519_sign(
        signature.data_ptr<uint8_t>(),
        message_contig.data_ptr<uint8_t>(),
        message.numel(),
        private_key_contig.data_ptr<uint8_t>()
    );

    return signature;
}

// Verify Ed25519 signature
// public_key: 1D tensor of 32 bytes
// message: 1D tensor of arbitrary length
// signature: 1D tensor of 64 bytes
// Returns: scalar bool tensor (true if valid, false otherwise)
at::Tensor ed25519_verify(
    const at::Tensor& public_key,
    const at::Tensor& message,
    const at::Tensor& signature
) {
    TORCH_CHECK(public_key.dim() == 1 && public_key.size(0) == 32,
        "ed25519_verify: public_key must be a 1D tensor of 32 bytes, got shape ", public_key.sizes());
    TORCH_CHECK(public_key.dtype() == at::kByte,
        "ed25519_verify: public_key must be uint8, got ", public_key.dtype());
    TORCH_CHECK(message.dim() == 1,
        "ed25519_verify: message must be a 1D tensor, got ", message.dim(), " dimensions");
    TORCH_CHECK(message.dtype() == at::kByte,
        "ed25519_verify: message must be uint8, got ", message.dtype());
    TORCH_CHECK(signature.dim() == 1 && signature.size(0) == 64,
        "ed25519_verify: signature must be a 1D tensor of 64 bytes, got shape ", signature.sizes());
    TORCH_CHECK(signature.dtype() == at::kByte,
        "ed25519_verify: signature must be uint8, got ", signature.dtype());

    auto public_key_contig = public_key.contiguous();
    auto message_contig = message.contiguous();
    auto signature_contig = signature.contiguous();

    bool valid = kernel::encryption::ed25519_verify(
        signature_contig.data_ptr<uint8_t>(),
        message_contig.data_ptr<uint8_t>(),
        message.numel(),
        public_key_contig.data_ptr<uint8_t>()
    );

    return at::tensor(valid ? 1 : 0, at::kByte);
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("ed25519_keypair", &ed25519_keypair);
    m.impl("ed25519_sign", &ed25519_sign);
    m.impl("ed25519_verify", &ed25519_verify);
}

}  // namespace torchscience::cpu::encryption
