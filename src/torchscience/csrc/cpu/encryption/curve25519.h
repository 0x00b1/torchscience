#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/curve25519.h"

namespace torchscience::cpu::encryption {

// X25519 scalar multiplication: output = scalar * point
// scalar: 1D tensor of 32 bytes (private key)
// point: 1D tensor of 32 bytes (public key x-coordinate)
// Returns: 1D tensor of 32 bytes (shared secret)
at::Tensor x25519(const at::Tensor& scalar, const at::Tensor& point) {
    TORCH_CHECK(scalar.dim() == 1 && scalar.size(0) == 32,
        "x25519: scalar must be a 1D tensor of 32 bytes, got shape ", scalar.sizes());
    TORCH_CHECK(scalar.dtype() == at::kByte,
        "x25519: scalar must be uint8, got ", scalar.dtype());
    TORCH_CHECK(point.dim() == 1 && point.size(0) == 32,
        "x25519: point must be a 1D tensor of 32 bytes, got shape ", point.sizes());
    TORCH_CHECK(point.dtype() == at::kByte,
        "x25519: point must be uint8, got ", point.dtype());

    auto scalar_contig = scalar.contiguous();
    auto point_contig = point.contiguous();

    auto output = at::empty({32}, scalar.options());

    kernel::encryption::x25519_scalarmult(
        output.data_ptr<uint8_t>(),
        scalar_contig.data_ptr<uint8_t>(),
        point_contig.data_ptr<uint8_t>()
    );

    return output;
}

// X25519 base point multiplication: output = scalar * G
// scalar: 1D tensor of 32 bytes (private key)
// Returns: 1D tensor of 32 bytes (public key x-coordinate)
at::Tensor x25519_base(const at::Tensor& scalar) {
    TORCH_CHECK(scalar.dim() == 1 && scalar.size(0) == 32,
        "x25519_base: scalar must be a 1D tensor of 32 bytes, got shape ", scalar.sizes());
    TORCH_CHECK(scalar.dtype() == at::kByte,
        "x25519_base: scalar must be uint8, got ", scalar.dtype());

    auto scalar_contig = scalar.contiguous();

    auto output = at::empty({32}, scalar.options());

    kernel::encryption::x25519_scalarmult_base(
        output.data_ptr<uint8_t>(),
        scalar_contig.data_ptr<uint8_t>()
    );

    return output;
}

// Generate X25519 keypair: (private_key, public_key)
// seed: 1D tensor of 32 bytes (random seed for private key)
// Returns: tuple of (private_key, public_key), each 32 bytes
std::tuple<at::Tensor, at::Tensor> x25519_keypair(const at::Tensor& seed) {
    TORCH_CHECK(seed.dim() == 1 && seed.size(0) == 32,
        "x25519_keypair: seed must be a 1D tensor of 32 bytes, got shape ", seed.sizes());
    TORCH_CHECK(seed.dtype() == at::kByte,
        "x25519_keypair: seed must be uint8, got ", seed.dtype());

    // Private key is the seed (clamping happens inside x25519_scalarmult_base)
    auto private_key = seed.contiguous().clone();

    // Public key = private_key * G
    auto public_key = x25519_base(private_key);

    return std::make_tuple(private_key, public_key);
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("x25519", &x25519);
    m.impl("x25519_base", &x25519_base);
    m.impl("x25519_keypair", &x25519_keypair);
}

}  // namespace torchscience::cpu::encryption
