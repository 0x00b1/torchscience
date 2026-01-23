#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/hkdf.h"

namespace torchscience::cpu::encryption {

// HKDF-Extract: PRK = HMAC-SHA256(salt, IKM)
// Extracts a pseudorandom key (PRK) from input keying material (IKM)
// salt: 1D tensor of bytes (optional salt value)
// ikm: 1D tensor of bytes (input keying material)
// Returns: 32-byte PRK tensor
at::Tensor hkdf_extract_sha256(const at::Tensor& salt, const at::Tensor& ikm) {
    TORCH_CHECK(salt.dim() == 1, "hkdf_extract_sha256: salt must be 1D");
    TORCH_CHECK(salt.dtype() == at::kByte, "hkdf_extract_sha256: salt must be uint8");
    TORCH_CHECK(ikm.dim() == 1, "hkdf_extract_sha256: ikm must be 1D");
    TORCH_CHECK(ikm.dtype() == at::kByte, "hkdf_extract_sha256: ikm must be uint8");

    auto salt_contig = salt.contiguous();
    auto ikm_contig = ikm.contiguous();

    // PRK is always 32 bytes (SHA256 output)
    auto prk = at::empty({32}, ikm.options());

    kernel::encryption::hkdf_extract_sha256(
        prk.data_ptr<uint8_t>(),
        salt_contig.data_ptr<uint8_t>(),
        salt.numel(),
        ikm_contig.data_ptr<uint8_t>(),
        ikm.numel()
    );

    return prk;
}

// HKDF-Expand: OKM = expand(PRK, info, L)
// Expands the pseudorandom key (PRK) into output keying material (OKM)
// prk: 32-byte PRK tensor from HKDF-Extract
// info: 1D tensor of bytes (context and application specific information)
// output_len: desired output length in bytes (max 255 * 32 = 8160 bytes)
// Returns: output_len-byte OKM tensor
at::Tensor hkdf_expand_sha256(const at::Tensor& prk, const at::Tensor& info, int64_t output_len) {
    TORCH_CHECK(prk.dim() == 1, "hkdf_expand_sha256: prk must be 1D");
    TORCH_CHECK(prk.dtype() == at::kByte, "hkdf_expand_sha256: prk must be uint8");
    TORCH_CHECK(prk.numel() == 32, "hkdf_expand_sha256: prk must be 32 bytes");
    TORCH_CHECK(info.dim() == 1, "hkdf_expand_sha256: info must be 1D");
    TORCH_CHECK(info.dtype() == at::kByte, "hkdf_expand_sha256: info must be uint8");
    TORCH_CHECK(output_len > 0, "hkdf_expand_sha256: output_len must be positive");
    TORCH_CHECK(output_len <= 255 * 32, "hkdf_expand_sha256: output_len must be at most 8160 bytes");

    auto prk_contig = prk.contiguous();
    auto info_contig = info.contiguous();

    auto okm = at::empty({output_len}, prk.options());

    kernel::encryption::hkdf_expand_sha256(
        okm.data_ptr<uint8_t>(),
        output_len,
        prk_contig.data_ptr<uint8_t>(),
        info_contig.data_ptr<uint8_t>(),
        info.numel()
    );

    return okm;
}

// Combined HKDF (extract + expand)
// Derives output keying material (OKM) from input keying material (IKM)
// using HKDF-SHA256 as specified in RFC 5869
// ikm: 1D tensor of bytes (input keying material)
// salt: 1D tensor of bytes (optional salt value)
// info: 1D tensor of bytes (context and application specific information)
// output_len: desired output length in bytes (max 255 * 32 = 8160 bytes)
// Returns: output_len-byte OKM tensor
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

    auto ikm_contig = ikm.contiguous();
    auto salt_contig = salt.contiguous();
    auto info_contig = info.contiguous();

    auto okm = at::empty({output_len}, ikm.options());

    kernel::encryption::hkdf_sha256(
        okm.data_ptr<uint8_t>(),
        output_len,
        ikm_contig.data_ptr<uint8_t>(),
        ikm.numel(),
        salt_contig.data_ptr<uint8_t>(),
        salt.numel(),
        info_contig.data_ptr<uint8_t>(),
        info.numel()
    );

    return okm;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("hkdf_extract_sha256", &hkdf_extract_sha256);
    m.impl("hkdf_expand_sha256", &hkdf_expand_sha256);
    m.impl("hkdf_sha256", &hkdf_sha256);
}

}  // namespace torchscience::cpu::encryption
