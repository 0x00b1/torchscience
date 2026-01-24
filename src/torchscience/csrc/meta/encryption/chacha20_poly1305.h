#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::encryption {

// ChaCha20-Poly1305 AEAD encryption meta implementation (shape inference only)
std::tuple<at::Tensor, at::Tensor> chacha20_poly1305_encrypt(
    const at::Tensor& plaintext,
    const at::Tensor& key,
    const at::Tensor& nonce,
    const at::Tensor& aad
) {
    TORCH_CHECK(plaintext.dim() == 1,
        "chacha20_poly1305_encrypt: plaintext must be a 1D tensor");
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20_poly1305_encrypt: key must be a 1D tensor of 32 bytes");
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20_poly1305_encrypt: nonce must be a 1D tensor of 12 bytes");
    TORCH_CHECK(aad.dim() == 1,
        "chacha20_poly1305_encrypt: aad must be a 1D tensor");

    auto ciphertext = at::empty({plaintext.size(0)}, plaintext.options());
    auto tag = at::empty({16}, plaintext.options());
    return std::make_tuple(ciphertext, tag);
}

// ChaCha20-Poly1305 AEAD decryption meta implementation (shape inference only)
at::Tensor chacha20_poly1305_decrypt(
    const at::Tensor& ciphertext,
    const at::Tensor& key,
    const at::Tensor& nonce,
    const at::Tensor& aad,
    const at::Tensor& tag
) {
    TORCH_CHECK(ciphertext.dim() == 1,
        "chacha20_poly1305_decrypt: ciphertext must be a 1D tensor");
    TORCH_CHECK(key.dim() == 1 && key.size(0) == 32,
        "chacha20_poly1305_decrypt: key must be a 1D tensor of 32 bytes");
    TORCH_CHECK(nonce.dim() == 1 && nonce.size(0) == 12,
        "chacha20_poly1305_decrypt: nonce must be a 1D tensor of 12 bytes");
    TORCH_CHECK(aad.dim() == 1,
        "chacha20_poly1305_decrypt: aad must be a 1D tensor");
    TORCH_CHECK(tag.dim() == 1 && tag.size(0) == 16,
        "chacha20_poly1305_decrypt: tag must be a 1D tensor of 16 bytes");

    return at::empty({ciphertext.size(0)}, ciphertext.options());
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
    m.impl("chacha20_poly1305_encrypt", &chacha20_poly1305_encrypt);
    m.impl("chacha20_poly1305_decrypt", &chacha20_poly1305_decrypt);
}

}  // namespace torchscience::meta::encryption
