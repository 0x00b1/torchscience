// src/torchscience/csrc/kernel/encryption/poly1305.h
#pragma once

#include <cstdint>
#include <cstring>

namespace torchscience::kernel::encryption {

// Poly1305 MAC implementation
// Produces a 16-byte authentication tag from a message and 32-byte key
// Uses 130-bit arithmetic modulo p = 2^130 - 5
//
// The 32-byte key is split into:
// - r (16 bytes): clamped and used as multiplier
// - s (16 bytes): added at the end
//
// For each 16-byte block: accumulator = (accumulator + block) * r mod p
// Final: tag = (accumulator + s) mod 2^128

// State using 5 x 26-bit limbs for 130-bit arithmetic
struct Poly1305State {
    uint32_t r[5];      // r after clamping, in 26-bit limbs
    uint32_t h[5];      // accumulator in 26-bit limbs
    uint32_t pad[4];    // s value (final addition) as 4 x 32-bit words
};

// Load 32-bit little-endian word from bytes
inline uint32_t poly1305_load_le32(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0])
         | (static_cast<uint32_t>(bytes[1]) << 8)
         | (static_cast<uint32_t>(bytes[2]) << 16)
         | (static_cast<uint32_t>(bytes[3]) << 24);
}

// Store 32-bit word as little-endian bytes
inline void poly1305_store_le32(uint8_t* bytes, uint32_t word) {
    bytes[0] = static_cast<uint8_t>(word);
    bytes[1] = static_cast<uint8_t>(word >> 8);
    bytes[2] = static_cast<uint8_t>(word >> 16);
    bytes[3] = static_cast<uint8_t>(word >> 24);
}

// Initialize Poly1305 state from 32-byte key
// key[0:16] = r (clamped), key[16:32] = s (pad)
inline void poly1305_init(Poly1305State& st, const uint8_t* key) {
    // Load r as little-endian 32-bit words
    uint32_t t0 = poly1305_load_le32(key + 0);
    uint32_t t1 = poly1305_load_le32(key + 4);
    uint32_t t2 = poly1305_load_le32(key + 8);
    uint32_t t3 = poly1305_load_le32(key + 12);

    // Clamp r:
    // - Clear top 4 bits of bytes 3, 7, 11, 15
    // - Clear bottom 2 bits of bytes 4, 8, 12
    // After clamping: r[3], r[7], r[11], r[15] & 0x0f
    //                 r[4], r[8], r[12] & 0xfc
    t0 &= 0x0fffffff;  // Clear top 4 bits of byte 3
    t1 &= 0x0ffffffc;  // Clear top 4 bits of byte 7, bottom 2 of byte 4
    t2 &= 0x0ffffffc;  // Clear top 4 bits of byte 11, bottom 2 of byte 8
    t3 &= 0x0ffffffc;  // Clear top 4 bits of byte 15, bottom 2 of byte 12

    // Convert clamped r to 5 x 26-bit limbs
    st.r[0] = t0 & 0x3ffffff;
    st.r[1] = ((t0 >> 26) | (t1 << 6)) & 0x3ffffff;
    st.r[2] = ((t1 >> 20) | (t2 << 12)) & 0x3ffffff;
    st.r[3] = ((t2 >> 14) | (t3 << 18)) & 0x3ffffff;
    st.r[4] = t3 >> 8;

    // Initialize accumulator to zero
    st.h[0] = 0;
    st.h[1] = 0;
    st.h[2] = 0;
    st.h[3] = 0;
    st.h[4] = 0;

    // Load s (pad) from key[16:32]
    st.pad[0] = poly1305_load_le32(key + 16);
    st.pad[1] = poly1305_load_le32(key + 20);
    st.pad[2] = poly1305_load_le32(key + 24);
    st.pad[3] = poly1305_load_le32(key + 28);
}

// Process 16-byte blocks
// hibit = 1 for full blocks, 0 for final partial block (already padded)
inline void poly1305_blocks(Poly1305State& st, const uint8_t* m, int64_t bytes, uint8_t hibit) {
    // Precompute r * 5 for faster modular reduction
    uint32_t r0 = st.r[0];
    uint32_t r1 = st.r[1];
    uint32_t r2 = st.r[2];
    uint32_t r3 = st.r[3];
    uint32_t r4 = st.r[4];

    uint32_t s1 = r1 * 5;
    uint32_t s2 = r2 * 5;
    uint32_t s3 = r3 * 5;
    uint32_t s4 = r4 * 5;

    uint32_t h0 = st.h[0];
    uint32_t h1 = st.h[1];
    uint32_t h2 = st.h[2];
    uint32_t h3 = st.h[3];
    uint32_t h4 = st.h[4];

    uint32_t hibit_val = static_cast<uint32_t>(hibit) << 24;

    while (bytes >= 16) {
        // Load 16-byte block as little-endian and convert to 5 limbs
        uint32_t t0 = poly1305_load_le32(m + 0);
        uint32_t t1 = poly1305_load_le32(m + 4);
        uint32_t t2 = poly1305_load_le32(m + 8);
        uint32_t t3 = poly1305_load_le32(m + 12);

        // Add block to accumulator
        h0 += t0 & 0x3ffffff;
        h1 += ((t0 >> 26) | (t1 << 6)) & 0x3ffffff;
        h2 += ((t1 >> 20) | (t2 << 12)) & 0x3ffffff;
        h3 += ((t2 >> 14) | (t3 << 18)) & 0x3ffffff;
        h4 += (t3 >> 8) | hibit_val;

        // Multiply h by r (mod 2^130 - 5)
        // Using 64-bit arithmetic to avoid overflow
        uint64_t d0 = static_cast<uint64_t>(h0) * r0
                    + static_cast<uint64_t>(h1) * s4
                    + static_cast<uint64_t>(h2) * s3
                    + static_cast<uint64_t>(h3) * s2
                    + static_cast<uint64_t>(h4) * s1;

        uint64_t d1 = static_cast<uint64_t>(h0) * r1
                    + static_cast<uint64_t>(h1) * r0
                    + static_cast<uint64_t>(h2) * s4
                    + static_cast<uint64_t>(h3) * s3
                    + static_cast<uint64_t>(h4) * s2;

        uint64_t d2 = static_cast<uint64_t>(h0) * r2
                    + static_cast<uint64_t>(h1) * r1
                    + static_cast<uint64_t>(h2) * r0
                    + static_cast<uint64_t>(h3) * s4
                    + static_cast<uint64_t>(h4) * s3;

        uint64_t d3 = static_cast<uint64_t>(h0) * r3
                    + static_cast<uint64_t>(h1) * r2
                    + static_cast<uint64_t>(h2) * r1
                    + static_cast<uint64_t>(h3) * r0
                    + static_cast<uint64_t>(h4) * s4;

        uint64_t d4 = static_cast<uint64_t>(h0) * r4
                    + static_cast<uint64_t>(h1) * r3
                    + static_cast<uint64_t>(h2) * r2
                    + static_cast<uint64_t>(h3) * r1
                    + static_cast<uint64_t>(h4) * r0;

        // Partial reduction mod 2^130 - 5
        uint64_t c;
        c = d0 >> 26; h0 = static_cast<uint32_t>(d0) & 0x3ffffff; d1 += c;
        c = d1 >> 26; h1 = static_cast<uint32_t>(d1) & 0x3ffffff; d2 += c;
        c = d2 >> 26; h2 = static_cast<uint32_t>(d2) & 0x3ffffff; d3 += c;
        c = d3 >> 26; h3 = static_cast<uint32_t>(d3) & 0x3ffffff; d4 += c;
        c = d4 >> 26; h4 = static_cast<uint32_t>(d4) & 0x3ffffff;

        // Fold carry back: c * 5 (since 2^130 = 5 mod p)
        h0 += static_cast<uint32_t>(c) * 5;
        c = h0 >> 26; h0 &= 0x3ffffff;
        h1 += static_cast<uint32_t>(c);

        m += 16;
        bytes -= 16;
    }

    st.h[0] = h0;
    st.h[1] = h1;
    st.h[2] = h2;
    st.h[3] = h3;
    st.h[4] = h4;
}

// Finalize and compute 16-byte tag
inline void poly1305_finish(Poly1305State& st, uint8_t* tag) {
    uint32_t h0 = st.h[0];
    uint32_t h1 = st.h[1];
    uint32_t h2 = st.h[2];
    uint32_t h3 = st.h[3];
    uint32_t h4 = st.h[4];

    // Full carry chain
    uint32_t c;
    c = h1 >> 26; h1 &= 0x3ffffff; h2 += c;
    c = h2 >> 26; h2 &= 0x3ffffff; h3 += c;
    c = h3 >> 26; h3 &= 0x3ffffff; h4 += c;
    c = h4 >> 26; h4 &= 0x3ffffff; h0 += c * 5;
    c = h0 >> 26; h0 &= 0x3ffffff; h1 += c;

    // Compute h + (-p) = h - (2^130 - 5) = h - 2^130 + 5
    // If h >= p, the result is h - p, otherwise h
    uint32_t g0 = h0 + 5;
    c = g0 >> 26; g0 &= 0x3ffffff;
    uint32_t g1 = h1 + c;
    c = g1 >> 26; g1 &= 0x3ffffff;
    uint32_t g2 = h2 + c;
    c = g2 >> 26; g2 &= 0x3ffffff;
    uint32_t g3 = h3 + c;
    c = g3 >> 26; g3 &= 0x3ffffff;
    uint32_t g4 = h4 + c - (1 << 26);

    // Select h if h < p (g4 has bit 31 set), else g
    uint32_t mask = (g4 >> 31) - 1;  // All 1s if g4 >= 0, all 0s if g4 < 0
    g0 &= mask;
    g1 &= mask;
    g2 &= mask;
    g3 &= mask;
    g4 &= mask;
    mask = ~mask;
    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;
    h4 = (h4 & mask) | g4;

    // Convert from 5 x 26-bit limbs to 4 x 32-bit words
    uint32_t f0 = h0 | (h1 << 26);
    uint32_t f1 = (h1 >> 6) | (h2 << 20);
    uint32_t f2 = (h2 >> 12) | (h3 << 14);
    uint32_t f3 = (h3 >> 18) | (h4 << 8);

    // Add pad (s) and output tag
    uint64_t t;
    t = static_cast<uint64_t>(f0) + st.pad[0];
    poly1305_store_le32(tag + 0, static_cast<uint32_t>(t));
    t = static_cast<uint64_t>(f1) + st.pad[1] + (t >> 32);
    poly1305_store_le32(tag + 4, static_cast<uint32_t>(t));
    t = static_cast<uint64_t>(f2) + st.pad[2] + (t >> 32);
    poly1305_store_le32(tag + 8, static_cast<uint32_t>(t));
    t = static_cast<uint64_t>(f3) + st.pad[3] + (t >> 32);
    poly1305_store_le32(tag + 12, static_cast<uint32_t>(t));
}

// Main Poly1305 function: compute 16-byte tag from message and 32-byte key
inline void poly1305(uint8_t* tag, const uint8_t* msg, int64_t msg_len, const uint8_t* key) {
    Poly1305State st;
    poly1305_init(st, key);

    // Process full 16-byte blocks
    int64_t full_blocks = msg_len / 16;
    if (full_blocks > 0) {
        poly1305_blocks(st, msg, full_blocks * 16, 1);
    }

    // Process final partial block if any
    int64_t remaining = msg_len % 16;
    if (remaining > 0) {
        uint8_t final_block[16] = {0};
        std::memcpy(final_block, msg + full_blocks * 16, remaining);
        final_block[remaining] = 1;  // Append 0x01 byte (padding)
        poly1305_blocks(st, final_block, 16, 0);
    }

    poly1305_finish(st, tag);
}

// Verify a Poly1305 tag (constant-time comparison)
inline bool poly1305_verify(
    const uint8_t* tag,
    const uint8_t* msg,
    int64_t msg_len,
    const uint8_t* key
) {
    uint8_t computed_tag[16];
    poly1305(computed_tag, msg, msg_len, key);

    // Constant-time comparison to prevent timing attacks
    uint8_t diff = 0;
    for (int i = 0; i < 16; i++) {
        diff |= tag[i] ^ computed_tag[i];
    }
    return diff == 0;
}

}  // namespace torchscience::kernel::encryption
