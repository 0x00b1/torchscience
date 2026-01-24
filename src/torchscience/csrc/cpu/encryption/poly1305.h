#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/poly1305.h"

namespace torchscience::cpu::encryption {

// Poly1305 one-time authenticator
// data: 1D tensor of arbitrary length (message to authenticate)
// key: 1D tensor of 32 bytes (one-time key)
// Returns: 1D tensor of 16 bytes (authentication tag)
at::Tensor poly1305(const at::Tensor& data, const at::Tensor& key) {
    TORCH_CHECK(data.dim() == 1,
        "poly1305: data must be a 1D tensor, got ", data.dim(), " dimensions");
    TORCH_CHECK(data.dtype() == at::kByte,
        "poly1305: data must be uint8, got ", data.dtype());
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "poly1305: key must be a 1D tensor of 32 bytes, got shape ", key.sizes());
    TORCH_CHECK(key.dtype() == at::kByte,
        "poly1305: key must be uint8, got ", key.dtype());

    auto data_contig = data.contiguous();
    auto key_contig = key.contiguous();

    auto output = at::empty({16}, data.options());

    kernel::encryption::poly1305(
        output.data_ptr<uint8_t>(),
        data_contig.data_ptr<uint8_t>(),
        data.size(0),
        key_contig.data_ptr<uint8_t>()
    );

    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("poly1305", &poly1305);
}

}  // namespace torchscience::cpu::encryption
