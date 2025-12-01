#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/sha256.h"

namespace torchscience::cpu::encryption {

at::Tensor sha256(const at::Tensor& data) {
    TORCH_CHECK(data.dtype() == at::kByte, "sha256: data must be uint8");
    auto data_contig = data.contiguous();
    auto output = at::empty({32}, data.options());
    kernel::encryption::sha256_hash(output.data_ptr<uint8_t>(), data_contig.data_ptr<uint8_t>(), data.numel());
    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) { m.impl("sha256", &sha256); }

}  // namespace torchscience::cpu::encryption
