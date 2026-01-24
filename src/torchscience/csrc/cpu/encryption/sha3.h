#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/keccak.h"

namespace torchscience::cpu::encryption {

at::Tensor sha3_256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "sha3_256: data must be uint8");
    auto data_contig = data.contiguous();
    auto output = at::empty({32}, data.options());
    kernel::encryption::sha3_256_hash(
        output.data_ptr<uint8_t>(),
        data_contig.data_ptr<uint8_t>(),
        data.numel()
    );
    return output;
}

at::Tensor sha3_512(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "sha3_512: data must be uint8");
    auto data_contig = data.contiguous();
    auto output = at::empty({64}, data.options());
    kernel::encryption::sha3_512_hash(
        output.data_ptr<uint8_t>(),
        data_contig.data_ptr<uint8_t>(),
        data.numel()
    );
    return output;
}

at::Tensor keccak256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "keccak256: data must be uint8");
    auto data_contig = data.contiguous();
    auto output = at::empty({32}, data.options());
    kernel::encryption::keccak256_hash(
        output.data_ptr<uint8_t>(),
        data_contig.data_ptr<uint8_t>(),
        data.numel()
    );
    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("sha3_256", &sha3_256);
    m.impl("sha3_512", &sha3_512);
    m.impl("keccak256", &keccak256);
}

}  // namespace torchscience::cpu::encryption
