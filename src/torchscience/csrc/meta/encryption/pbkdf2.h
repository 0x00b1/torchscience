#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// PBKDF2-SHA256 shape inference
// Returns a tensor of output_len bytes
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

    return at::empty({output_len}, password.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("pbkdf2_sha256", &pbkdf2_sha256);
}

}  // namespace torchscience::meta::encryption
