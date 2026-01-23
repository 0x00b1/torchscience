#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/aes.h"

namespace torchscience::cpu::encryption {

// AES block encryption (ECB mode, single 16-byte block)
// plaintext: 1D tensor of 16 bytes
// key: 1D tensor of 16 bytes (AES-128) or 32 bytes (AES-256)
// Returns: 1D tensor of 16 bytes (ciphertext)
at::Tensor aes_encrypt_block(const at::Tensor& plaintext, const at::Tensor& key) {
    TORCH_CHECK(plaintext.dim() == 1 && plaintext.size(0) == 16,
        "aes_encrypt_block: plaintext must be a 1D tensor of 16 bytes, got shape ", plaintext.sizes());
    TORCH_CHECK(plaintext.dtype() == at::kByte,
        "aes_encrypt_block: plaintext must be uint8, got ", plaintext.dtype());
    TORCH_CHECK(key.dim() == 1,
        "aes_encrypt_block: key must be a 1D tensor");
    TORCH_CHECK(key.size(0) == 16 || key.size(0) == 32,
        "aes_encrypt_block: key must be 16 bytes (AES-128) or 32 bytes (AES-256), got ", key.size(0));
    TORCH_CHECK(key.dtype() == at::kByte,
        "aes_encrypt_block: key must be uint8, got ", key.dtype());

    auto plaintext_contig = plaintext.contiguous();
    auto key_contig = key.contiguous();

    auto output = at::empty({16}, plaintext.options());

    kernel::encryption::aes_encrypt_block(
        output.data_ptr<uint8_t>(),
        plaintext_contig.data_ptr<uint8_t>(),
        key_contig.data_ptr<uint8_t>(),
        static_cast<int>(key.size(0))
    );

    return output;
}

// AES block decryption (ECB mode, single 16-byte block)
// ciphertext: 1D tensor of 16 bytes
// key: 1D tensor of 16 bytes (AES-128) or 32 bytes (AES-256)
// Returns: 1D tensor of 16 bytes (plaintext)
at::Tensor aes_decrypt_block(const at::Tensor& ciphertext, const at::Tensor& key) {
    TORCH_CHECK(ciphertext.dim() == 1 && ciphertext.size(0) == 16,
        "aes_decrypt_block: ciphertext must be a 1D tensor of 16 bytes, got shape ", ciphertext.sizes());
    TORCH_CHECK(ciphertext.dtype() == at::kByte,
        "aes_decrypt_block: ciphertext must be uint8, got ", ciphertext.dtype());
    TORCH_CHECK(key.dim() == 1,
        "aes_decrypt_block: key must be a 1D tensor");
    TORCH_CHECK(key.size(0) == 16 || key.size(0) == 32,
        "aes_decrypt_block: key must be 16 bytes (AES-128) or 32 bytes (AES-256), got ", key.size(0));
    TORCH_CHECK(key.dtype() == at::kByte,
        "aes_decrypt_block: key must be uint8, got ", key.dtype());

    auto ciphertext_contig = ciphertext.contiguous();
    auto key_contig = key.contiguous();

    auto output = at::empty({16}, ciphertext.options());

    kernel::encryption::aes_decrypt_block(
        output.data_ptr<uint8_t>(),
        ciphertext_contig.data_ptr<uint8_t>(),
        key_contig.data_ptr<uint8_t>(),
        static_cast<int>(key.size(0))
    );

    return output;
}

// AES-CTR mode encryption/decryption
// data: 1D tensor of arbitrary length (bytes to encrypt/decrypt)
// key: 1D tensor of 16 bytes (AES-128) or 32 bytes (AES-256)
// nonce: 1D tensor of 12 bytes
// counter: initial counter value (default 0)
// Returns: 1D tensor of same size as data (encrypted/decrypted result)
at::Tensor aes_ctr(
    const at::Tensor& data,
    const at::Tensor& key,
    const at::Tensor& nonce,
    int64_t counter
) {
    TORCH_CHECK(data.dim() == 1,
        "aes_ctr: data must be a 1D tensor, got ", data.dim(), " dimensions");
    TORCH_CHECK(data.dtype() == at::kByte,
        "aes_ctr: data must be uint8, got ", data.dtype());
    TORCH_CHECK(key.dim() == 1,
        "aes_ctr: key must be a 1D tensor");
    TORCH_CHECK(key.size(0) == 16 || key.size(0) == 32,
        "aes_ctr: key must be 16 bytes (AES-128) or 32 bytes (AES-256), got ", key.size(0));
    TORCH_CHECK(key.dtype() == at::kByte,
        "aes_ctr: key must be uint8, got ", key.dtype());
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "aes_ctr: nonce must be a 1D tensor of 12 bytes, got shape ", nonce.sizes());
    TORCH_CHECK(nonce.dtype() == at::kByte,
        "aes_ctr: nonce must be uint8, got ", nonce.dtype());
    TORCH_CHECK(counter >= 0,
        "aes_ctr: counter must be non-negative, got ", counter);

    auto data_contig = data.contiguous();
    auto key_contig = key.contiguous();
    auto nonce_contig = nonce.contiguous();

    auto output = at::empty({data.size(0)}, data.options());

    kernel::encryption::aes_ctr(
        output.data_ptr<uint8_t>(),
        data_contig.data_ptr<uint8_t>(),
        data.size(0),
        key_contig.data_ptr<uint8_t>(),
        static_cast<int>(key.size(0)),
        nonce_contig.data_ptr<uint8_t>(),
        static_cast<uint32_t>(counter)
    );

    return output;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("aes_encrypt_block", &aes_encrypt_block);
    m.impl("aes_decrypt_block", &aes_decrypt_block);
    m.impl("aes_ctr", &aes_ctr);
}

}  // namespace torchscience::cpu::encryption
