#pragma once

#include <cstdint>
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "sha256.h"

namespace torchscience::kernel::encryption {

// SHA256 output size in bytes
constexpr int64_t PBKDF2_SHA256_DIGEST_LEN = 32;
// SHA256 block size in bytes
constexpr int64_t PBKDF2_SHA256_BLOCK_SIZE = 64;

// HMAC-SHA256 for PBKDF2
// Computes HMAC-SHA256(key, data) and writes 32 bytes to output
inline void hmac_sha256(
    uint8_t* output,
    const uint8_t* key,
    int64_t key_len,
    const uint8_t* data,
    int64_t data_len
) {
    std::array<uint8_t, PBKDF2_SHA256_BLOCK_SIZE> k_pad = {};

    // If key is longer than block size, hash it first
    if (key_len > PBKDF2_SHA256_BLOCK_SIZE) {
        sha256_hash(k_pad.data(), key, key_len);
        // Remaining bytes are already zero
    } else {
        std::memcpy(k_pad.data(), key, key_len);
        // Remaining bytes are already zero
    }

    // Compute inner padding: k_pad XOR ipad (0x36)
    std::array<uint8_t, PBKDF2_SHA256_BLOCK_SIZE> inner_key;
    for (int64_t i = 0; i < PBKDF2_SHA256_BLOCK_SIZE; i++) {
        inner_key[i] = k_pad[i] ^ 0x36;
    }

    // Compute outer padding: k_pad XOR opad (0x5c)
    std::array<uint8_t, PBKDF2_SHA256_BLOCK_SIZE> outer_key;
    for (int64_t i = 0; i < PBKDF2_SHA256_BLOCK_SIZE; i++) {
        outer_key[i] = k_pad[i] ^ 0x5c;
    }

    // Inner hash: SHA256(inner_key || data)
    // Allocate buffer for inner_key || data
    std::vector<uint8_t> inner_input(PBKDF2_SHA256_BLOCK_SIZE + data_len);
    std::memcpy(inner_input.data(), inner_key.data(), PBKDF2_SHA256_BLOCK_SIZE);
    if (data_len > 0) {
        std::memcpy(inner_input.data() + PBKDF2_SHA256_BLOCK_SIZE, data, data_len);
    }

    std::array<uint8_t, PBKDF2_SHA256_DIGEST_LEN> inner_hash;
    sha256_hash(inner_hash.data(), inner_input.data(), inner_input.size());

    // Outer hash: SHA256(outer_key || inner_hash)
    std::array<uint8_t, PBKDF2_SHA256_BLOCK_SIZE + PBKDF2_SHA256_DIGEST_LEN> outer_input;
    std::memcpy(outer_input.data(), outer_key.data(), PBKDF2_SHA256_BLOCK_SIZE);
    std::memcpy(outer_input.data() + PBKDF2_SHA256_BLOCK_SIZE, inner_hash.data(), PBKDF2_SHA256_DIGEST_LEN);

    sha256_hash(output, outer_input.data(), outer_input.size());
}

// PBKDF2-HMAC-SHA256
// Derives a key from password and salt using the PBKDF2 algorithm (RFC 2898/8018)
//
// output: buffer to write derived key (must be at least output_len bytes)
// output_len: desired output length in bytes
// password: password bytes
// password_len: password length
// salt: salt bytes
// salt_len: salt length
// iterations: number of iterations (c)
inline void pbkdf2_sha256(
    uint8_t* output,
    int64_t output_len,
    const uint8_t* password,
    int64_t password_len,
    const uint8_t* salt,
    int64_t salt_len,
    int64_t iterations
) {
    // Number of blocks needed (each block produces 32 bytes)
    int64_t num_blocks = (output_len + PBKDF2_SHA256_DIGEST_LEN - 1) / PBKDF2_SHA256_DIGEST_LEN;

    // Allocate buffer for salt || INT(i)
    // INT(i) is a 4-byte big-endian encoding of block number (1-indexed)
    std::vector<uint8_t> salt_block(salt_len + 4);
    if (salt_len > 0) {
        std::memcpy(salt_block.data(), salt, salt_len);
    }

    int64_t output_offset = 0;

    for (int64_t block_idx = 1; block_idx <= num_blocks; block_idx++) {
        // Encode block index as 4-byte big-endian (1-indexed)
        salt_block[salt_len + 0] = static_cast<uint8_t>((block_idx >> 24) & 0xff);
        salt_block[salt_len + 1] = static_cast<uint8_t>((block_idx >> 16) & 0xff);
        salt_block[salt_len + 2] = static_cast<uint8_t>((block_idx >> 8) & 0xff);
        salt_block[salt_len + 3] = static_cast<uint8_t>(block_idx & 0xff);

        // U1 = PRF(Password, Salt || INT(i))
        std::array<uint8_t, PBKDF2_SHA256_DIGEST_LEN> u_prev;
        hmac_sha256(u_prev.data(), password, password_len, salt_block.data(), salt_len + 4);

        // T = U1 (start with U1 as the running XOR)
        std::array<uint8_t, PBKDF2_SHA256_DIGEST_LEN> t_block;
        std::memcpy(t_block.data(), u_prev.data(), PBKDF2_SHA256_DIGEST_LEN);

        // Uj = PRF(Password, Uj-1), T ^= Uj for j = 2..iterations
        for (int64_t j = 2; j <= iterations; j++) {
            std::array<uint8_t, PBKDF2_SHA256_DIGEST_LEN> u_curr;
            hmac_sha256(u_curr.data(), password, password_len, u_prev.data(), PBKDF2_SHA256_DIGEST_LEN);

            // XOR into T
            for (int64_t k = 0; k < PBKDF2_SHA256_DIGEST_LEN; k++) {
                t_block[k] ^= u_curr[k];
            }

            // U_prev = U_curr for next iteration
            std::memcpy(u_prev.data(), u_curr.data(), PBKDF2_SHA256_DIGEST_LEN);
        }

        // Copy T_block to output (may be partial for last block)
        int64_t bytes_to_copy = std::min(PBKDF2_SHA256_DIGEST_LEN, output_len - output_offset);
        std::memcpy(output + output_offset, t_block.data(), bytes_to_copy);
        output_offset += bytes_to_copy;
    }
}

}  // namespace torchscience::kernel::encryption
