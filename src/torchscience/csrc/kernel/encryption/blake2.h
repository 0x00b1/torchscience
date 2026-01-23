#pragma once

#include <cstdint>
#include <array>
#include <cstring>

namespace torchscience::kernel::encryption {

// BLAKE2b initialization vector (first 64 bits of fractional parts of sqrt of first 8 primes)
constexpr std::array<uint64_t, 8> BLAKE2B_IV = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// BLAKE2s initialization vector (first 32 bits of fractional parts of sqrt of first 8 primes)
constexpr std::array<uint32_t, 8> BLAKE2S_IV = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
};

// Sigma permutation schedule (same for both BLAKE2b and BLAKE2s)
constexpr std::array<std::array<uint8_t, 16>, 10> BLAKE2_SIGMA = {{
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
    {14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3},
    {11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4},
    { 7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8},
    { 9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13},
    { 2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9},
    {12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11},
    {13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10},
    { 6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5},
    {10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0}
}};

// BLAKE2b rotation constants: 32, 24, 16, 63
inline uint64_t blake2b_rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// BLAKE2b G mixing function
inline void blake2b_g(uint64_t* v, int a, int b, int c, int d, uint64_t x, uint64_t y) {
    v[a] = v[a] + v[b] + x;
    v[d] = blake2b_rotr64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = blake2b_rotr64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + y;
    v[d] = blake2b_rotr64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = blake2b_rotr64(v[b] ^ v[c], 63);
}

// BLAKE2b compression function
inline void blake2b_compress(
    uint64_t* h,
    const uint8_t* block,
    uint64_t t0,
    uint64_t t1,
    bool is_last
) {
    // Initialize working vector
    uint64_t v[16];
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = BLAKE2B_IV[i];
    }

    // XOR in offset counters
    v[12] ^= t0;
    v[13] ^= t1;

    // XOR in finalization flag
    if (is_last) {
        v[14] = ~v[14];
    }

    // Get message words
    uint64_t m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = 0;
        for (int j = 0; j < 8; j++) {
            m[i] |= static_cast<uint64_t>(block[i * 8 + j]) << (j * 8);
        }
    }

    // 12 rounds of mixing
    for (int round = 0; round < 12; round++) {
        const auto& s = BLAKE2_SIGMA[round % 10];

        blake2b_g(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        blake2b_g(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        blake2b_g(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        blake2b_g(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        blake2b_g(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        blake2b_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake2b_g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        blake2b_g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    // Finalize
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

// BLAKE2b hash function
inline void blake2b_hash(
    uint8_t* output,
    const uint8_t* input,
    int64_t input_len,
    const uint8_t* key,
    int64_t key_len,
    int64_t digest_len
) {
    // Initialize state with IV
    uint64_t h[8];
    for (int i = 0; i < 8; i++) {
        h[i] = BLAKE2B_IV[i];
    }

    // XOR in parameter block
    // Parameter block: digest_len in byte 0, key_len in byte 1, fanout=1 in byte 2, depth=1 in byte 3
    h[0] ^= 0x01010000ULL ^ (static_cast<uint64_t>(key_len) << 8) ^ static_cast<uint64_t>(digest_len);

    uint64_t t0 = 0;  // bytes compressed counter (low)
    uint64_t t1 = 0;  // bytes compressed counter (high)

    // If key is provided, process it as first block
    std::array<uint8_t, 128> block = {};
    if (key_len > 0) {
        std::memcpy(block.data(), key, key_len);
        t0 = 128;
        blake2b_compress(h, block.data(), t0, t1, input_len == 0);
    }

    // Process input in 128-byte blocks
    int64_t offset = 0;
    while (offset + 128 <= input_len) {
        t0 += 128;
        if (t0 < 128) t1++;  // Handle overflow
        blake2b_compress(h, input + offset, t0, t1, false);
        offset += 128;
    }

    // Process final block
    block = {};
    int64_t remaining = input_len - offset;
    if (remaining > 0) {
        std::memcpy(block.data(), input + offset, remaining);
    }

    // Update counter for final block
    if (key_len > 0 && input_len == 0) {
        // Key-only case already handled
    } else {
        t0 += remaining;
        if (t0 < static_cast<uint64_t>(remaining)) t1++;  // Handle overflow
        blake2b_compress(h, block.data(), t0, t1, true);
    }

    // Extract output
    for (int64_t i = 0; i < digest_len; i++) {
        output[i] = static_cast<uint8_t>(h[i / 8] >> ((i % 8) * 8));
    }
}

// BLAKE2s rotation constants: 16, 12, 8, 7
inline uint32_t blake2s_rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// BLAKE2s G mixing function
inline void blake2s_g(uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y) {
    v[a] = v[a] + v[b] + x;
    v[d] = blake2s_rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = blake2s_rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + y;
    v[d] = blake2s_rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = blake2s_rotr32(v[b] ^ v[c], 7);
}

// BLAKE2s compression function
inline void blake2s_compress(
    uint32_t* h,
    const uint8_t* block,
    uint32_t t0,
    uint32_t t1,
    bool is_last
) {
    // Initialize working vector
    uint32_t v[16];
    for (int i = 0; i < 8; i++) {
        v[i] = h[i];
        v[i + 8] = BLAKE2S_IV[i];
    }

    // XOR in offset counters
    v[12] ^= t0;
    v[13] ^= t1;

    // XOR in finalization flag
    if (is_last) {
        v[14] = ~v[14];
    }

    // Get message words
    uint32_t m[16];
    for (int i = 0; i < 16; i++) {
        m[i] = 0;
        for (int j = 0; j < 4; j++) {
            m[i] |= static_cast<uint32_t>(block[i * 4 + j]) << (j * 8);
        }
    }

    // 10 rounds of mixing
    for (int round = 0; round < 10; round++) {
        const auto& s = BLAKE2_SIGMA[round];

        blake2s_g(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        blake2s_g(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        blake2s_g(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        blake2s_g(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        blake2s_g(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        blake2s_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake2s_g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        blake2s_g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    // Finalize
    for (int i = 0; i < 8; i++) {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

// BLAKE2s hash function
inline void blake2s_hash(
    uint8_t* output,
    const uint8_t* input,
    int64_t input_len,
    const uint8_t* key,
    int64_t key_len,
    int64_t digest_len
) {
    // Initialize state with IV
    uint32_t h[8];
    for (int i = 0; i < 8; i++) {
        h[i] = BLAKE2S_IV[i];
    }

    // XOR in parameter block
    // Parameter block: digest_len in byte 0, key_len in byte 1, fanout=1 in byte 2, depth=1 in byte 3
    h[0] ^= 0x01010000U ^ (static_cast<uint32_t>(key_len) << 8) ^ static_cast<uint32_t>(digest_len);

    uint32_t t0 = 0;  // bytes compressed counter (low)
    uint32_t t1 = 0;  // bytes compressed counter (high)

    // If key is provided, process it as first block
    std::array<uint8_t, 64> block = {};
    if (key_len > 0) {
        std::memcpy(block.data(), key, key_len);
        t0 = 64;
        blake2s_compress(h, block.data(), t0, t1, input_len == 0);
    }

    // Process input in 64-byte blocks
    int64_t offset = 0;
    while (offset + 64 <= input_len) {
        t0 += 64;
        if (t0 < 64) t1++;  // Handle overflow
        blake2s_compress(h, input + offset, t0, t1, false);
        offset += 64;
    }

    // Process final block
    block = {};
    int64_t remaining = input_len - offset;
    if (remaining > 0) {
        std::memcpy(block.data(), input + offset, remaining);
    }

    // Update counter for final block
    if (key_len > 0 && input_len == 0) {
        // Key-only case already handled
    } else {
        t0 += remaining;
        if (t0 < static_cast<uint32_t>(remaining)) t1++;  // Handle overflow
        blake2s_compress(h, block.data(), t0, t1, true);
    }

    // Extract output
    for (int64_t i = 0; i < digest_len; i++) {
        output[i] = static_cast<uint8_t>(h[i / 4] >> ((i % 4) * 8));
    }
}

}  // namespace torchscience::kernel::encryption
