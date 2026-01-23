#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/pbkdf2.h"

namespace torchscience::cpu::encryption {

// PBKDF2-SHA256
// password: 1D tensor of bytes
// salt: 1D tensor of bytes
// iterations: number of iterations
// output_len: desired output length
at::Tensor pbkdf2_sha256(
    const at::Tensor& password,
    const at::Tensor& salt,
    int64_t iterations,
    int64_t output_len
) {
    TORCH_CHECK(password.dim() == 1, "pbkdf2_sha256: password must be 1D");
    TORCH_CHECK(password.dtype() == at::kByte, "pbkdf2_sha256: password must be uint8");
    TORCH_CHECK(salt.dim() == 1, "pbkdf2_sha256: salt must be 1D");
    TORCH_CHECK(salt.dtype() == at::kByte, "pbkdf2_sha256: salt must be uint8");
    TORCH_CHECK(iterations > 0, "pbkdf2_sha256: iterations must be positive");
    TORCH_CHECK(output_len > 0, "pbkdf2_sha256: output_len must be positive");

    auto password_contig = password.contiguous();
    auto salt_contig = salt.contiguous();

    auto output = at::empty({output_len}, password.options());

    kernel::encryption::pbkdf2_sha256(
        output.data_ptr<uint8_t>(),
        output_len,
        password_contig.data_ptr<uint8_t>(),
        password.numel(),
        salt_contig.data_ptr<uint8_t>(),
        salt.numel(),
        iterations
    );

    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("pbkdf2_sha256", &pbkdf2_sha256);
}

}  // namespace torchscience::cpu::encryption
