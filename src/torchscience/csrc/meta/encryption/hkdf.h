#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// HKDF-Extract shape inference
// Returns a 32-byte PRK tensor (SHA256 output)
at::Tensor hkdf_extract_sha256(const at::Tensor& salt, const at::Tensor& ikm) {
    TORCH_CHECK(salt.dim() == 1, "hkdf_extract_sha256: salt must be 1D");
    TORCH_CHECK(salt.dtype() == at::kByte, "hkdf_extract_sha256: salt must be uint8");
    TORCH_CHECK(ikm.dim() == 1, "hkdf_extract_sha256: ikm must be 1D");
    TORCH_CHECK(ikm.dtype() == at::kByte, "hkdf_extract_sha256: ikm must be uint8");

    return at::empty({32}, ikm.options());
}

// HKDF-Expand shape inference
// Returns a tensor of output_len bytes
at::Tensor hkdf_expand_sha256(const at::Tensor& prk, const at::Tensor& info, int64_t output_len) {
    TORCH_CHECK(prk.dim() == 1, "hkdf_expand_sha256: prk must be 1D");
    TORCH_CHECK(prk.dtype() == at::kByte, "hkdf_expand_sha256: prk must be uint8");
    TORCH_CHECK(prk.numel() == 32, "hkdf_expand_sha256: prk must be 32 bytes");
    TORCH_CHECK(info.dim() == 1, "hkdf_expand_sha256: info must be 1D");
    TORCH_CHECK(info.dtype() == at::kByte, "hkdf_expand_sha256: info must be uint8");
    TORCH_CHECK(output_len > 0, "hkdf_expand_sha256: output_len must be positive");
    TORCH_CHECK(output_len <= 255 * 32, "hkdf_expand_sha256: output_len must be at most 8160 bytes");

    return at::empty({output_len}, prk.options());
}

// Combined HKDF shape inference
// Returns a tensor of output_len bytes
at::Tensor hkdf_sha256(
    const at::Tensor& ikm,
    const at::Tensor& salt,
    const at::Tensor& info,
    int64_t output_len
) {
    TORCH_CHECK(ikm.dim() == 1, "hkdf_sha256: ikm must be 1D");
    TORCH_CHECK(ikm.dtype() == at::kByte, "hkdf_sha256: ikm must be uint8");
    TORCH_CHECK(salt.dim() == 1, "hkdf_sha256: salt must be 1D");
    TORCH_CHECK(salt.dtype() == at::kByte, "hkdf_sha256: salt must be uint8");
    TORCH_CHECK(info.dim() == 1, "hkdf_sha256: info must be 1D");
    TORCH_CHECK(info.dtype() == at::kByte, "hkdf_sha256: info must be uint8");
    TORCH_CHECK(output_len > 0, "hkdf_sha256: output_len must be positive");
    TORCH_CHECK(output_len <= 255 * 32, "hkdf_sha256: output_len must be at most 8160 bytes");

    return at::empty({output_len}, ikm.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("hkdf_extract_sha256", &hkdf_extract_sha256);
    m.impl("hkdf_expand_sha256", &hkdf_expand_sha256);
    m.impl("hkdf_sha256", &hkdf_sha256);
}

}  // namespace torchscience::meta::encryption
