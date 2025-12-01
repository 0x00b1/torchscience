#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "../../kernel/encryption/chacha20.h"

namespace torchscience::cpu::encryption {

at::Tensor chacha20(
    const at::Tensor& key,
    const at::Tensor& nonce,
    int64_t num_bytes,
    int64_t counter
) {
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20: key must be a 1D tensor of 32 bytes, got shape ", key.sizes());
    TORCH_CHECK(key.dtype() == at::kByte,
        "chacha20: key must be uint8, got ", key.dtype());
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20: nonce must be a 1D tensor of 12 bytes, got shape ", nonce.sizes());
    TORCH_CHECK(nonce.dtype() == at::kByte,
        "chacha20: nonce must be uint8, got ", nonce.dtype());
    TORCH_CHECK(num_bytes > 0,
        "chacha20: num_bytes must be positive, got ", num_bytes);
    TORCH_CHECK(counter >= 0,
        "chacha20: counter must be non-negative, got ", counter);

    auto key_contig = key.contiguous();
    auto nonce_contig = nonce.contiguous();

    auto output = at::empty({num_bytes}, key.options().dtype(at::kByte));

    kernel::encryption::chacha20_keystream(
        output.data_ptr<uint8_t>(),
        num_bytes,
        key_contig.data_ptr<uint8_t>(),
        nonce_contig.data_ptr<uint8_t>(),
        static_cast<uint32_t>(counter)
    );

    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("chacha20", &chacha20);
}

}  // namespace torchscience::cpu::encryption
