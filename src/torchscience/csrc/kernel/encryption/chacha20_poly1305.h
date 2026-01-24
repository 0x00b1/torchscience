// src/torchscience/csrc/kernel/encryption/chacha20_poly1305.h
#pragma once

#include "chacha20.h"
#include "poly1305.h"

namespace torchscience::kernel::encryption {

// ChaCha20-Poly1305 AEAD (Authenticated Encryption with Associated Data)
// RFC 8439 construction
//
// Encryption:
// 1. Generate Poly1305 one-time key: ChaCha20(key, nonce, counter=0)[0:32]
// 2. Encrypt plaintext with ChaCha20(key, nonce, counter=1)
// 3. Construct Poly1305 input:
//    - AAD || pad to 16 bytes
//    - ciphertext || pad to 16 bytes
//    - len(AAD) as 8-byte little-endian
//    - len(ciphertext) as 8-byte little-endian
// 4. Compute tag = Poly1305(poly_key, poly_input)

// Store 64-bit value as little-endian bytes
inline void store_le64(uint8_t* bytes, uint64_t val) {
    bytes[0] = static_cast<uint8_t>(val);
    bytes[1] = static_cast<uint8_t>(val >> 8);
    bytes[2] = static_cast<uint8_t>(val >> 16);
    bytes[3] = static_cast<uint8_t>(val >> 24);
    bytes[4] = static_cast<uint8_t>(val >> 32);
    bytes[5] = static_cast<uint8_t>(val >> 40);
    bytes[6] = static_cast<uint8_t>(val >> 48);
    bytes[7] = static_cast<uint8_t>(val >> 56);
}

// Compute padding length to align to 16 bytes
inline int64_t pad16(int64_t len) {
    int64_t rem = len % 16;
    return rem == 0 ? 0 : 16 - rem;
}

// Internal: Compute Poly1305 tag over AEAD construction
inline void chacha20_poly1305_compute_tag(
    uint8_t* tag,
    const uint8_t* aad,
    int64_t aad_len,
    const uint8_t* ciphertext,
    int64_t ciphertext_len,
    const uint8_t* poly_key
) {
    Poly1305State st;
    poly1305_init(st, poly_key);

    // Process AAD
    if (aad_len > 0) {
        int64_t full_blocks = aad_len / 16;
        if (full_blocks > 0) {
            poly1305_blocks(st, aad, full_blocks * 16, 1);
        }
        int64_t remaining = aad_len % 16;
        if (remaining > 0) {
            uint8_t padded[16] = {0};
            std::memcpy(padded, aad + full_blocks * 16, remaining);
            poly1305_blocks(st, padded, 16, 1);
        }
    }

    // Process ciphertext
    if (ciphertext_len > 0) {
        int64_t full_blocks = ciphertext_len / 16;
        if (full_blocks > 0) {
            poly1305_blocks(st, ciphertext, full_blocks * 16, 1);
        }
        int64_t remaining = ciphertext_len % 16;
        if (remaining > 0) {
            uint8_t padded[16] = {0};
            std::memcpy(padded, ciphertext + full_blocks * 16, remaining);
            poly1305_blocks(st, padded, 16, 1);
        }
    }

    // Process lengths: aad_len || ciphertext_len (each 8 bytes, little-endian)
    uint8_t lengths[16];
    store_le64(lengths, static_cast<uint64_t>(aad_len));
    store_le64(lengths + 8, static_cast<uint64_t>(ciphertext_len));
    poly1305_blocks(st, lengths, 16, 1);

    poly1305_finish(st, tag);
}

// ChaCha20-Poly1305 AEAD encryption
// Inputs:
//   plaintext: message to encrypt (plaintext_len bytes)
//   aad: additional authenticated data (aad_len bytes, not encrypted but authenticated)
//   key: 32-byte encryption key
//   nonce: 12-byte nonce (must be unique per key)
// Outputs:
//   ciphertext: encrypted message (same size as plaintext)
//   tag: 16-byte authentication tag
inline void chacha20_poly1305_encrypt(
    uint8_t* ciphertext,
    uint8_t* tag,
    const uint8_t* plaintext,
    int64_t plaintext_len,
    const uint8_t* aad,
    int64_t aad_len,
    const uint8_t* key,
    const uint8_t* nonce
) {
    // Step 1: Generate Poly1305 one-time key using ChaCha20 with counter=0
    uint8_t poly_key[64];  // ChaCha20 produces 64 bytes per block, we use first 32
    chacha20_keystream(poly_key, 64, key, nonce, 0);

    // Step 2: Encrypt plaintext using ChaCha20 with counter=1
    if (plaintext_len > 0) {
        chacha20_keystream(ciphertext, plaintext_len, key, nonce, 1);
        for (int64_t i = 0; i < plaintext_len; i++) {
            ciphertext[i] ^= plaintext[i];
        }
    }

    // Step 3-4: Compute authentication tag
    chacha20_poly1305_compute_tag(tag, aad, aad_len, ciphertext, plaintext_len, poly_key);
}

// ChaCha20-Poly1305 AEAD decryption
// Inputs:
//   ciphertext: encrypted message (ciphertext_len bytes)
//   aad: additional authenticated data (aad_len bytes)
//   key: 32-byte encryption key
//   nonce: 12-byte nonce
//   tag: 16-byte authentication tag to verify
// Outputs:
//   plaintext: decrypted message (same size as ciphertext)
// Returns:
//   true if authentication succeeded, false otherwise
//   On false return, plaintext contents are undefined (should be discarded)
inline bool chacha20_poly1305_decrypt(
    uint8_t* plaintext,
    const uint8_t* ciphertext,
    int64_t ciphertext_len,
    const uint8_t* aad,
    int64_t aad_len,
    const uint8_t* key,
    const uint8_t* nonce,
    const uint8_t* tag
) {
    // Step 1: Generate Poly1305 one-time key
    uint8_t poly_key[64];
    chacha20_keystream(poly_key, 64, key, nonce, 0);

    // Step 2: Verify authentication tag BEFORE decrypting
    uint8_t computed_tag[16];
    chacha20_poly1305_compute_tag(computed_tag, aad, aad_len, ciphertext, ciphertext_len, poly_key);

    // Constant-time tag comparison to prevent timing attacks
    uint8_t diff = 0;
    for (int i = 0; i < 16; i++) {
        diff |= tag[i] ^ computed_tag[i];
    }

    if (diff != 0) {
        // Authentication failed - zero out plaintext and return false
        std::memset(plaintext, 0, ciphertext_len);
        return false;
    }

    // Step 3: Decrypt ciphertext using ChaCha20 with counter=1
    if (ciphertext_len > 0) {
        chacha20_keystream(plaintext, ciphertext_len, key, nonce, 1);
        for (int64_t i = 0; i < ciphertext_len; i++) {
            plaintext[i] ^= ciphertext[i];
        }
    }

    return true;
}

// In-place ChaCha20-Poly1305 encryption
// The plaintext buffer is overwritten with ciphertext
inline void chacha20_poly1305_encrypt_inplace(
    uint8_t* data,           // plaintext on input, ciphertext on output
    int64_t data_len,
    uint8_t* tag,            // 16-byte output tag
    const uint8_t* aad,
    int64_t aad_len,
    const uint8_t* key,
    const uint8_t* nonce
) {
    // Generate Poly1305 key
    uint8_t poly_key[64];
    chacha20_keystream(poly_key, 64, key, nonce, 0);

    // Encrypt in place
    if (data_len > 0) {
        uint8_t keystream[64];
        uint32_t counter = 1;
        int64_t offset = 0;

        while (offset < data_len) {
            int64_t block_len = std::min(static_cast<int64_t>(64), data_len - offset);
            chacha20_keystream(keystream, 64, key, nonce, counter);
            for (int64_t i = 0; i < block_len; i++) {
                data[offset + i] ^= keystream[i];
            }
            offset += 64;
            counter++;
        }
    }

    // Compute tag over ciphertext
    chacha20_poly1305_compute_tag(tag, aad, aad_len, data, data_len, poly_key);
}

// In-place ChaCha20-Poly1305 decryption
// The ciphertext buffer is overwritten with plaintext if authentication succeeds
inline bool chacha20_poly1305_decrypt_inplace(
    uint8_t* data,           // ciphertext on input, plaintext on output (if verified)
    int64_t data_len,
    const uint8_t* aad,
    int64_t aad_len,
    const uint8_t* key,
    const uint8_t* nonce,
    const uint8_t* tag
) {
    // Generate Poly1305 key
    uint8_t poly_key[64];
    chacha20_keystream(poly_key, 64, key, nonce, 0);

    // Verify tag FIRST
    uint8_t computed_tag[16];
    chacha20_poly1305_compute_tag(computed_tag, aad, aad_len, data, data_len, poly_key);

    uint8_t diff = 0;
    for (int i = 0; i < 16; i++) {
        diff |= tag[i] ^ computed_tag[i];
    }

    if (diff != 0) {
        std::memset(data, 0, data_len);
        return false;
    }

    // Decrypt in place
    if (data_len > 0) {
        uint8_t keystream[64];
        uint32_t counter = 1;
        int64_t offset = 0;

        while (offset < data_len) {
            int64_t block_len = std::min(static_cast<int64_t>(64), data_len - offset);
            chacha20_keystream(keystream, 64, key, nonce, counter);
            for (int64_t i = 0; i < block_len; i++) {
                data[offset + i] ^= keystream[i];
            }
            offset += 64;
            counter++;
        }
    }

    return true;
}

}  // namespace torchscience::kernel::encryption
