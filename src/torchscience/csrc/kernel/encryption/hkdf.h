#pragma once

#include <cstdint>
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "sha256.h"

namespace torchscience::kernel::encryption {

// SHA256 output size in bytes (HashLen)
constexpr int64_t HKDF_SHA256_HASH_LEN = 32;
// SHA256 block size in bytes
constexpr int64_t HKDF_SHA256_BLOCK_SIZE = 64;
// Maximum output length for HKDF-SHA256: 255 * HashLen = 8160 bytes
constexpr int64_t HKDF_SHA256_MAX_OUTPUT_LEN = 255 * HKDF_SHA256_HASH_LEN;

// HMAC-SHA256 for HKDF
// Computes HMAC-SHA256(key, data) and writes 32 bytes to output
inline void hkdf_hmac_sha256(
    uint8_t* output,
    const uint8_t* key,
    int64_t key_len,
    const uint8_t* data,
    int64_t data_len
) {
    std::array<uint8_t, HKDF_SHA256_BLOCK_SIZE> k_pad = {};

    // If key is longer than block size, hash it first
    if (key_len > HKDF_SHA256_BLOCK_SIZE) {
        sha256_hash(k_pad.data(), key, key_len);
        // Remaining bytes are already zero
    } else {
        std::memcpy(k_pad.data(), key, key_len);
        // Remaining bytes are already zero
    }

    // Compute inner padding: k_pad XOR ipad (0x36)
    std::array<uint8_t, HKDF_SHA256_BLOCK_SIZE> inner_key;
    for (int64_t i = 0; i < HKDF_SHA256_BLOCK_SIZE; i++) {
        inner_key[i] = k_pad[i] ^ 0x36;
    }

    // Compute outer padding: k_pad XOR opad (0x5c)
    std::array<uint8_t, HKDF_SHA256_BLOCK_SIZE> outer_key;
    for (int64_t i = 0; i < HKDF_SHA256_BLOCK_SIZE; i++) {
        outer_key[i] = k_pad[i] ^ 0x5c;
    }

    // Inner hash: SHA256(inner_key || data)
    std::vector<uint8_t> inner_input(HKDF_SHA256_BLOCK_SIZE + data_len);
    std::memcpy(inner_input.data(), inner_key.data(), HKDF_SHA256_BLOCK_SIZE);
    if (data_len > 0) {
        std::memcpy(inner_input.data() + HKDF_SHA256_BLOCK_SIZE, data, data_len);
    }

    std::array<uint8_t, HKDF_SHA256_HASH_LEN> inner_hash;
    sha256_hash(inner_hash.data(), inner_input.data(), inner_input.size());

    // Outer hash: SHA256(outer_key || inner_hash)
    std::array<uint8_t, HKDF_SHA256_BLOCK_SIZE + HKDF_SHA256_HASH_LEN> outer_input;
    std::memcpy(outer_input.data(), outer_key.data(), HKDF_SHA256_BLOCK_SIZE);
    std::memcpy(outer_input.data() + HKDF_SHA256_BLOCK_SIZE, inner_hash.data(), HKDF_SHA256_HASH_LEN);

    sha256_hash(output, outer_input.data(), outer_input.size());
}

// HKDF-Extract: PRK = HMAC-SHA256(salt, IKM)
// Extracts a pseudorandom key (PRK) from input keying material (IKM)
//
// prk: 32-byte output buffer for the pseudorandom key
// salt: optional salt value (if NULL or salt_len == 0, uses 32 zero bytes)
// salt_len: length of salt in bytes
// ikm: input keying material
// ikm_len: length of IKM in bytes
inline void hkdf_extract_sha256(
    uint8_t* prk,
    const uint8_t* salt,
    int64_t salt_len,
    const uint8_t* ikm,
    int64_t ikm_len
) {
    // If salt is not provided, use a string of HashLen zeros
    if (salt == nullptr || salt_len == 0) {
        std::array<uint8_t, HKDF_SHA256_HASH_LEN> default_salt = {};
        hkdf_hmac_sha256(prk, default_salt.data(), HKDF_SHA256_HASH_LEN, ikm, ikm_len);
    } else {
        hkdf_hmac_sha256(prk, salt, salt_len, ikm, ikm_len);
    }
}

// HKDF-Expand: OKM = expand(PRK, info, L)
// Expands the pseudorandom key (PRK) into output keying material (OKM)
//
// okm: output buffer for the output keying material
// okm_len: desired output length in bytes (max 255 * 32 = 8160 bytes)
// prk: 32-byte pseudorandom key from HKDF-Extract
// info: optional context and application specific information
// info_len: length of info in bytes
inline void hkdf_expand_sha256(
    uint8_t* okm,
    int64_t okm_len,
    const uint8_t* prk,
    const uint8_t* info,
    int64_t info_len
) {
    // Number of blocks needed: N = ceil(L/HashLen)
    int64_t num_blocks = (okm_len + HKDF_SHA256_HASH_LEN - 1) / HKDF_SHA256_HASH_LEN;

    // T(0) = empty string (zero length)
    std::array<uint8_t, HKDF_SHA256_HASH_LEN> t_prev = {};
    int64_t t_prev_len = 0;

    int64_t output_offset = 0;

    for (int64_t i = 1; i <= num_blocks; i++) {
        // T(i) = HMAC-Hash(PRK, T(i-1) || info || counter)
        // where counter is a single byte (0x01 to 0xFF)

        // Build input: T(i-1) || info || counter
        int64_t input_len = t_prev_len + info_len + 1;
        std::vector<uint8_t> input(input_len);

        int64_t pos = 0;

        // Copy T(i-1) (empty for first iteration)
        if (t_prev_len > 0) {
            std::memcpy(input.data() + pos, t_prev.data(), t_prev_len);
            pos += t_prev_len;
        }

        // Copy info
        if (info_len > 0 && info != nullptr) {
            std::memcpy(input.data() + pos, info, info_len);
            pos += info_len;
        }

        // Append counter byte (1-indexed)
        input[pos] = static_cast<uint8_t>(i);

        // Compute T(i) = HMAC-SHA256(PRK, input)
        std::array<uint8_t, HKDF_SHA256_HASH_LEN> t_curr;
        hkdf_hmac_sha256(t_curr.data(), prk, HKDF_SHA256_HASH_LEN, input.data(), input_len);

        // Copy to output (may be partial for last block)
        int64_t bytes_to_copy = std::min(HKDF_SHA256_HASH_LEN, okm_len - output_offset);
        std::memcpy(okm + output_offset, t_curr.data(), bytes_to_copy);
        output_offset += bytes_to_copy;

        // Save T(i) for next iteration
        std::memcpy(t_prev.data(), t_curr.data(), HKDF_SHA256_HASH_LEN);
        t_prev_len = HKDF_SHA256_HASH_LEN;
    }
}

// Combined HKDF (extract + expand)
// Derives output keying material (OKM) from input keying material (IKM)
// using HKDF-SHA256 as specified in RFC 5869
//
// okm: output buffer for the output keying material
// okm_len: desired output length in bytes (max 255 * 32 = 8160 bytes)
// ikm: input keying material
// ikm_len: length of IKM in bytes
// salt: optional salt value (if NULL or salt_len == 0, uses 32 zero bytes)
// salt_len: length of salt in bytes
// info: optional context and application specific information
// info_len: length of info in bytes
inline void hkdf_sha256(
    uint8_t* okm,
    int64_t okm_len,
    const uint8_t* ikm,
    int64_t ikm_len,
    const uint8_t* salt,
    int64_t salt_len,
    const uint8_t* info,
    int64_t info_len
) {
    // Step 1: Extract - PRK = HMAC-Hash(salt, IKM)
    std::array<uint8_t, HKDF_SHA256_HASH_LEN> prk;
    hkdf_extract_sha256(prk.data(), salt, salt_len, ikm, ikm_len);

    // Step 2: Expand - OKM = HKDF-Expand(PRK, info, L)
    hkdf_expand_sha256(okm, okm_len, prk.data(), info, info_len);
}

}  // namespace torchscience::kernel::encryption
