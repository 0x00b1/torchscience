#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>
#include "../../kernel/encryption/chacha20_poly1305.h"

namespace torchscience::cpu::encryption {

// ChaCha20-Poly1305 AEAD encryption
// plaintext: 1D tensor of arbitrary length (message to encrypt)
// key: 1D tensor of 32 bytes (encryption key)
// nonce: 1D tensor of 12 bytes (unique per key)
// aad: 1D tensor of arbitrary length (additional authenticated data, not encrypted)
// Returns: tuple of (ciphertext, tag)
//   ciphertext: 1D tensor of same size as plaintext
//   tag: 1D tensor of 16 bytes (authentication tag)
std::tuple<at::Tensor, at::Tensor> chacha20_poly1305_encrypt(
    const at::Tensor& plaintext,
    const at::Tensor& key,
    const at::Tensor& nonce,
    const at::Tensor& aad
) {
    TORCH_CHECK(plaintext.dim() == 1,
        "chacha20_poly1305_encrypt: plaintext must be a 1D tensor, got ", plaintext.dim(), " dimensions");
    TORCH_CHECK(plaintext.dtype() == at::kByte,
        "chacha20_poly1305_encrypt: plaintext must be uint8, got ", plaintext.dtype());
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20_poly1305_encrypt: key must be a 1D tensor of 32 bytes, got shape ", key.sizes());
    TORCH_CHECK(key.dtype() == at::kByte,
        "chacha20_poly1305_encrypt: key must be uint8, got ", key.dtype());
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20_poly1305_encrypt: nonce must be a 1D tensor of 12 bytes, got shape ", nonce.sizes());
    TORCH_CHECK(nonce.dtype() == at::kByte,
        "chacha20_poly1305_encrypt: nonce must be uint8, got ", nonce.dtype());
    TORCH_CHECK(aad.dim() == 1,
        "chacha20_poly1305_encrypt: aad must be a 1D tensor, got ", aad.dim(), " dimensions");
    TORCH_CHECK(aad.dtype() == at::kByte,
        "chacha20_poly1305_encrypt: aad must be uint8, got ", aad.dtype());

    auto plaintext_contig = plaintext.contiguous();
    auto key_contig = key.contiguous();
    auto nonce_contig = nonce.contiguous();
    auto aad_contig = aad.contiguous();

    auto ciphertext = at::empty({plaintext.size(0)}, plaintext.options());
    auto tag = at::empty({16}, plaintext.options());

    kernel::encryption::chacha20_poly1305_encrypt(
        ciphertext.data_ptr<uint8_t>(),
        tag.data_ptr<uint8_t>(),
        plaintext_contig.data_ptr<uint8_t>(),
        plaintext.size(0),
        aad_contig.data_ptr<uint8_t>(),
        aad.size(0),
        key_contig.data_ptr<uint8_t>(),
        nonce_contig.data_ptr<uint8_t>()
    );

    return std::make_tuple(ciphertext, tag);
}

// ChaCha20-Poly1305 AEAD decryption
// ciphertext: 1D tensor of arbitrary length (encrypted message)
// key: 1D tensor of 32 bytes (encryption key)
// nonce: 1D tensor of 12 bytes (must match encryption nonce)
// aad: 1D tensor of arbitrary length (must match encryption aad)
// tag: 1D tensor of 16 bytes (authentication tag)
// Returns: plaintext (1D tensor of same size as ciphertext)
// Throws: if authentication fails
at::Tensor chacha20_poly1305_decrypt(
    const at::Tensor& ciphertext,
    const at::Tensor& key,
    const at::Tensor& nonce,
    const at::Tensor& aad,
    const at::Tensor& tag
) {
    TORCH_CHECK(ciphertext.dim() == 1,
        "chacha20_poly1305_decrypt: ciphertext must be a 1D tensor, got ", ciphertext.dim(), " dimensions");
    TORCH_CHECK(ciphertext.dtype() == at::kByte,
        "chacha20_poly1305_decrypt: ciphertext must be uint8, got ", ciphertext.dtype());
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20_poly1305_decrypt: key must be a 1D tensor of 32 bytes, got shape ", key.sizes());
    TORCH_CHECK(key.dtype() == at::kByte,
        "chacha20_poly1305_decrypt: key must be uint8, got ", key.dtype());
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20_poly1305_decrypt: nonce must be a 1D tensor of 12 bytes, got shape ", nonce.sizes());
    TORCH_CHECK(nonce.dtype() == at::kByte,
        "chacha20_poly1305_decrypt: nonce must be uint8, got ", nonce.dtype());
    TORCH_CHECK(aad.dim() == 1,
        "chacha20_poly1305_decrypt: aad must be a 1D tensor, got ", aad.dim(), " dimensions");
    TORCH_CHECK(aad.dtype() == at::kByte,
        "chacha20_poly1305_decrypt: aad must be uint8, got ", aad.dtype());
    TORCH_CHECK(tag.dim() == 1 && tag.size(0) == 16,
        "chacha20_poly1305_decrypt: tag must be a 1D tensor of 16 bytes, got shape ", tag.sizes());
    TORCH_CHECK(tag.dtype() == at::kByte,
        "chacha20_poly1305_decrypt: tag must be uint8, got ", tag.dtype());

    auto ciphertext_contig = ciphertext.contiguous();
    auto key_contig = key.contiguous();
    auto nonce_contig = nonce.contiguous();
    auto aad_contig = aad.contiguous();
    auto tag_contig = tag.contiguous();

    auto plaintext = at::empty({ciphertext.size(0)}, ciphertext.options());

    bool success = kernel::encryption::chacha20_poly1305_decrypt(
        plaintext.data_ptr<uint8_t>(),
        ciphertext_contig.data_ptr<uint8_t>(),
        ciphertext.size(0),
        aad_contig.data_ptr<uint8_t>(),
        aad.size(0),
        key_contig.data_ptr<uint8_t>(),
        nonce_contig.data_ptr<uint8_t>(),
        tag_contig.data_ptr<uint8_t>()
    );

    TORCH_CHECK(success, "chacha20_poly1305_decrypt: authentication failed");

    return plaintext;
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("chacha20_poly1305_encrypt", &chacha20_poly1305_encrypt);
    m.impl("chacha20_poly1305_decrypt", &chacha20_poly1305_decrypt);
}

}  // namespace torchscience::cpu::encryption
