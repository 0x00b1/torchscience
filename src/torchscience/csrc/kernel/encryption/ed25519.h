// src/torchscience/csrc/kernel/encryption/ed25519.h
#pragma once

#include "curve25519.h"
#include <cstring>

namespace torchscience::kernel::encryption {

// ============================================================================
// SHA-512 Implementation (required for Ed25519 signatures per RFC 8032)
// ============================================================================

// SHA-512 round constants (first 64 bits of fractional parts of cube roots of first 80 primes)
constexpr uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

inline uint64_t sha512_rotr(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

inline uint64_t sha512_ch(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (~x & z);
}

inline uint64_t sha512_maj(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

inline uint64_t sha512_sigma0(uint64_t x) {
    return sha512_rotr(x, 28) ^ sha512_rotr(x, 34) ^ sha512_rotr(x, 39);
}

inline uint64_t sha512_sigma1(uint64_t x) {
    return sha512_rotr(x, 14) ^ sha512_rotr(x, 18) ^ sha512_rotr(x, 41);
}

inline uint64_t sha512_gamma0(uint64_t x) {
    return sha512_rotr(x, 1) ^ sha512_rotr(x, 8) ^ (x >> 7);
}

inline uint64_t sha512_gamma1(uint64_t x) {
    return sha512_rotr(x, 19) ^ sha512_rotr(x, 61) ^ (x >> 6);
}

inline void sha512_transform(uint64_t* state, const uint8_t* block) {
    uint64_t w[80];

    // Load message block (big-endian)
    for (int i = 0; i < 16; i++) {
        w[i] = (static_cast<uint64_t>(block[i * 8]) << 56)
             | (static_cast<uint64_t>(block[i * 8 + 1]) << 48)
             | (static_cast<uint64_t>(block[i * 8 + 2]) << 40)
             | (static_cast<uint64_t>(block[i * 8 + 3]) << 32)
             | (static_cast<uint64_t>(block[i * 8 + 4]) << 24)
             | (static_cast<uint64_t>(block[i * 8 + 5]) << 16)
             | (static_cast<uint64_t>(block[i * 8 + 6]) << 8)
             | static_cast<uint64_t>(block[i * 8 + 7]);
    }

    // Extend to 80 words
    for (int i = 16; i < 80; i++) {
        w[i] = sha512_gamma1(w[i - 2]) + w[i - 7] + sha512_gamma0(w[i - 15]) + w[i - 16];
    }

    // Initialize working variables
    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];

    // Main loop
    for (int i = 0; i < 80; i++) {
        uint64_t t1 = h + sha512_sigma1(e) + sha512_ch(e, f, g) + SHA512_K[i] + w[i];
        uint64_t t2 = sha512_sigma0(a) + sha512_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Update state
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// SHA-512 hash function: produces 64-byte output
inline void sha512_hash(uint8_t* output, const uint8_t* input, int64_t input_len) {
    // Initial hash values (first 64 bits of fractional parts of sqrt of first 8 primes)
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    // Process complete 128-byte blocks
    int64_t num_blocks = input_len / 128;
    for (int64_t i = 0; i < num_blocks; i++) {
        sha512_transform(state, input + i * 128);
    }

    // Prepare final block(s) with padding
    uint8_t final_blocks[256];
    std::memset(final_blocks, 0, 256);
    int64_t remaining = input_len % 128;
    std::memcpy(final_blocks, input + num_blocks * 128, remaining);

    // Append 1 bit
    final_blocks[remaining] = 0x80;

    // Determine number of padding blocks needed
    int pad_blocks = (remaining < 112) ? 1 : 2;

    // Append length in bits (big-endian, 128-bit length but we only use low 64 bits)
    uint64_t bit_len = static_cast<uint64_t>(input_len) * 8;
    int len_offset = pad_blocks * 128 - 8;
    for (int i = 0; i < 8; i++) {
        final_blocks[len_offset + i] = static_cast<uint8_t>(bit_len >> (56 - i * 8));
    }

    // Process final blocks
    for (int i = 0; i < pad_blocks; i++) {
        sha512_transform(state, final_blocks + i * 128);
    }

    // Output hash (big-endian)
    for (int i = 0; i < 8; i++) {
        output[i * 8] = static_cast<uint8_t>(state[i] >> 56);
        output[i * 8 + 1] = static_cast<uint8_t>(state[i] >> 48);
        output[i * 8 + 2] = static_cast<uint8_t>(state[i] >> 40);
        output[i * 8 + 3] = static_cast<uint8_t>(state[i] >> 32);
        output[i * 8 + 4] = static_cast<uint8_t>(state[i] >> 24);
        output[i * 8 + 5] = static_cast<uint8_t>(state[i] >> 16);
        output[i * 8 + 6] = static_cast<uint8_t>(state[i] >> 8);
        output[i * 8 + 7] = static_cast<uint8_t>(state[i]);
    }
}

// SHA-512 incremental context for streaming hashing
struct Sha512Context {
    uint64_t state[8];
    uint8_t buffer[128];
    int64_t buffer_len;
    int64_t total_len;
};

inline void sha512_init(Sha512Context& ctx) {
    ctx.state[0] = 0x6a09e667f3bcc908ULL;
    ctx.state[1] = 0xbb67ae8584caa73bULL;
    ctx.state[2] = 0x3c6ef372fe94f82bULL;
    ctx.state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx.state[4] = 0x510e527fade682d1ULL;
    ctx.state[5] = 0x9b05688c2b3e6c1fULL;
    ctx.state[6] = 0x1f83d9abfb41bd6bULL;
    ctx.state[7] = 0x5be0cd19137e2179ULL;
    ctx.buffer_len = 0;
    ctx.total_len = 0;
}

inline void sha512_update(Sha512Context& ctx, const uint8_t* data, int64_t len) {
    ctx.total_len += len;

    // If we have buffered data, try to fill the buffer first
    if (ctx.buffer_len > 0) {
        int64_t space = 128 - ctx.buffer_len;
        int64_t to_copy = (len < space) ? len : space;
        std::memcpy(ctx.buffer + ctx.buffer_len, data, to_copy);
        ctx.buffer_len += to_copy;
        data += to_copy;
        len -= to_copy;

        if (ctx.buffer_len == 128) {
            sha512_transform(ctx.state, ctx.buffer);
            ctx.buffer_len = 0;
        }
    }

    // Process complete blocks directly from input
    while (len >= 128) {
        sha512_transform(ctx.state, data);
        data += 128;
        len -= 128;
    }

    // Buffer remaining data
    if (len > 0) {
        std::memcpy(ctx.buffer, data, len);
        ctx.buffer_len = len;
    }
}

inline void sha512_final(Sha512Context& ctx, uint8_t* output) {
    // Pad the message
    uint8_t final_blocks[256];
    std::memset(final_blocks, 0, 256);
    std::memcpy(final_blocks, ctx.buffer, ctx.buffer_len);
    final_blocks[ctx.buffer_len] = 0x80;

    int pad_blocks = (ctx.buffer_len < 112) ? 1 : 2;
    uint64_t bit_len = static_cast<uint64_t>(ctx.total_len) * 8;
    int len_offset = pad_blocks * 128 - 8;
    for (int i = 0; i < 8; i++) {
        final_blocks[len_offset + i] = static_cast<uint8_t>(bit_len >> (56 - i * 8));
    }

    for (int i = 0; i < pad_blocks; i++) {
        sha512_transform(ctx.state, final_blocks + i * 128);
    }

    // Output hash
    for (int i = 0; i < 8; i++) {
        output[i * 8] = static_cast<uint8_t>(ctx.state[i] >> 56);
        output[i * 8 + 1] = static_cast<uint8_t>(ctx.state[i] >> 48);
        output[i * 8 + 2] = static_cast<uint8_t>(ctx.state[i] >> 40);
        output[i * 8 + 3] = static_cast<uint8_t>(ctx.state[i] >> 32);
        output[i * 8 + 4] = static_cast<uint8_t>(ctx.state[i] >> 24);
        output[i * 8 + 5] = static_cast<uint8_t>(ctx.state[i] >> 16);
        output[i * 8 + 6] = static_cast<uint8_t>(ctx.state[i] >> 8);
        output[i * 8 + 7] = static_cast<uint8_t>(ctx.state[i]);
    }
}

// ============================================================================
// Scalar Arithmetic mod L (Ed25519 curve order)
// ============================================================================
//
// L = 2^252 + 27742317777372353535851937790883648493
// L in little-endian bytes:
// ed d3 f5 5c 1a 63 12 58 d6 9c f7 a2 de f9 de 14
// 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10

// Load 3 bytes as little-endian uint64 (for sc25519 operations)
inline uint64_t sc25519_load3(const uint8_t* b) {
    return static_cast<uint64_t>(b[0])
         | (static_cast<uint64_t>(b[1]) << 8)
         | (static_cast<uint64_t>(b[2]) << 16);
}

// Load 4 bytes as little-endian uint64 (for sc25519 operations)
inline uint64_t sc25519_load4(const uint8_t* b) {
    return static_cast<uint64_t>(b[0])
         | (static_cast<uint64_t>(b[1]) << 8)
         | (static_cast<uint64_t>(b[2]) << 16)
         | (static_cast<uint64_t>(b[3]) << 24);
}


// Reduce a 64-byte (512-bit) value modulo L
// Input: 64 bytes (little-endian)
// Output: 32 bytes (little-endian), reduced mod L
//
// Uses the identity: 2^252 ≡ -27742317777372353535851937790883648493 (mod L)
// which allows efficient reduction of high bits
inline void sc25519_reduce(uint8_t* s) {
    // Load the 64-byte input as 24 21-bit limbs
    // This allows schoolbook multiplication without overflow in int64_t
    // Uses load3/load4 to avoid reading beyond the 64-byte buffer

    int64_t s0  = 0x1fffff & sc25519_load3(s);
    int64_t s1  = 0x1fffff & (sc25519_load4(s + 2) >> 5);
    int64_t s2  = 0x1fffff & (sc25519_load3(s + 5) >> 2);
    int64_t s3  = 0x1fffff & (sc25519_load4(s + 7) >> 7);
    int64_t s4  = 0x1fffff & (sc25519_load4(s + 10) >> 4);
    int64_t s5  = 0x1fffff & (sc25519_load3(s + 13) >> 1);
    int64_t s6  = 0x1fffff & (sc25519_load4(s + 15) >> 6);
    int64_t s7  = 0x1fffff & (sc25519_load3(s + 18) >> 3);
    int64_t s8  = 0x1fffff & sc25519_load3(s + 21);
    int64_t s9  = 0x1fffff & (sc25519_load4(s + 23) >> 5);
    int64_t s10 = 0x1fffff & (sc25519_load3(s + 26) >> 2);
    int64_t s11 = 0x1fffff & (sc25519_load4(s + 28) >> 7);
    int64_t s12 = 0x1fffff & (sc25519_load4(s + 31) >> 4);
    int64_t s13 = 0x1fffff & (sc25519_load3(s + 34) >> 1);
    int64_t s14 = 0x1fffff & (sc25519_load4(s + 36) >> 6);
    int64_t s15 = 0x1fffff & (sc25519_load3(s + 39) >> 3);
    int64_t s16 = 0x1fffff & sc25519_load3(s + 42);
    int64_t s17 = 0x1fffff & (sc25519_load4(s + 44) >> 5);
    int64_t s18 = 0x1fffff & (sc25519_load3(s + 47) >> 2);
    int64_t s19 = 0x1fffff & (sc25519_load4(s + 49) >> 7);
    int64_t s20 = 0x1fffff & (sc25519_load4(s + 52) >> 4);
    int64_t s21 = 0x1fffff & (sc25519_load3(s + 55) >> 1);
    int64_t s22 = 0x1fffff & (sc25519_load4(s + 57) >> 6);
    int64_t s23 = (sc25519_load4(s + 60) >> 3);

    int64_t carry;

    // Reduce using: 2^252 ≡ -mu (mod L)
    // where mu = 27742317777372353535851937790883648493
    // mu in limb form for reduction operations

    s11 += s23 * 666643;
    s12 += s23 * 470296;
    s13 += s23 * 654183;
    s14 -= s23 * 997805;
    s15 += s23 * 136657;
    s16 -= s23 * 683901;
    s23 = 0;

    s10 += s22 * 666643;
    s11 += s22 * 470296;
    s12 += s22 * 654183;
    s13 -= s22 * 997805;
    s14 += s22 * 136657;
    s15 -= s22 * 683901;
    s22 = 0;

    s9 += s21 * 666643;
    s10 += s21 * 470296;
    s11 += s21 * 654183;
    s12 -= s21 * 997805;
    s13 += s21 * 136657;
    s14 -= s21 * 683901;
    s21 = 0;

    s8 += s20 * 666643;
    s9 += s20 * 470296;
    s10 += s20 * 654183;
    s11 -= s20 * 997805;
    s12 += s20 * 136657;
    s13 -= s20 * 683901;
    s20 = 0;

    s7 += s19 * 666643;
    s8 += s19 * 470296;
    s9 += s19 * 654183;
    s10 -= s19 * 997805;
    s11 += s19 * 136657;
    s12 -= s19 * 683901;
    s19 = 0;

    s6 += s18 * 666643;
    s7 += s18 * 470296;
    s8 += s18 * 654183;
    s9 -= s18 * 997805;
    s10 += s18 * 136657;
    s11 -= s18 * 683901;
    s18 = 0;

    // Carry propagation
    carry = (s6 + (1 << 20)) >> 21; s7 += carry; s6 -= carry << 21;
    carry = (s8 + (1 << 20)) >> 21; s9 += carry; s8 -= carry << 21;
    carry = (s10 + (1 << 20)) >> 21; s11 += carry; s10 -= carry << 21;
    carry = (s12 + (1 << 20)) >> 21; s13 += carry; s12 -= carry << 21;
    carry = (s14 + (1 << 20)) >> 21; s15 += carry; s14 -= carry << 21;
    carry = (s16 + (1 << 20)) >> 21; s17 += carry; s16 -= carry << 21;

    carry = (s7 + (1 << 20)) >> 21; s8 += carry; s7 -= carry << 21;
    carry = (s9 + (1 << 20)) >> 21; s10 += carry; s9 -= carry << 21;
    carry = (s11 + (1 << 20)) >> 21; s12 += carry; s11 -= carry << 21;
    carry = (s13 + (1 << 20)) >> 21; s14 += carry; s13 -= carry << 21;
    carry = (s15 + (1 << 20)) >> 21; s16 += carry; s15 -= carry << 21;

    s5 += s17 * 666643;
    s6 += s17 * 470296;
    s7 += s17 * 654183;
    s8 -= s17 * 997805;
    s9 += s17 * 136657;
    s10 -= s17 * 683901;
    s17 = 0;

    s4 += s16 * 666643;
    s5 += s16 * 470296;
    s6 += s16 * 654183;
    s7 -= s16 * 997805;
    s8 += s16 * 136657;
    s9 -= s16 * 683901;
    s16 = 0;

    s3 += s15 * 666643;
    s4 += s15 * 470296;
    s5 += s15 * 654183;
    s6 -= s15 * 997805;
    s7 += s15 * 136657;
    s8 -= s15 * 683901;
    s15 = 0;

    s2 += s14 * 666643;
    s3 += s14 * 470296;
    s4 += s14 * 654183;
    s5 -= s14 * 997805;
    s6 += s14 * 136657;
    s7 -= s14 * 683901;
    s14 = 0;

    s1 += s13 * 666643;
    s2 += s13 * 470296;
    s3 += s13 * 654183;
    s4 -= s13 * 997805;
    s5 += s13 * 136657;
    s6 -= s13 * 683901;
    s13 = 0;

    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;

    // More carries
    carry = (s0 + (1 << 20)) >> 21; s1 += carry; s0 -= carry << 21;
    carry = (s2 + (1 << 20)) >> 21; s3 += carry; s2 -= carry << 21;
    carry = (s4 + (1 << 20)) >> 21; s5 += carry; s4 -= carry << 21;
    carry = (s6 + (1 << 20)) >> 21; s7 += carry; s6 -= carry << 21;
    carry = (s8 + (1 << 20)) >> 21; s9 += carry; s8 -= carry << 21;
    carry = (s10 + (1 << 20)) >> 21; s11 += carry; s10 -= carry << 21;

    carry = (s1 + (1 << 20)) >> 21; s2 += carry; s1 -= carry << 21;
    carry = (s3 + (1 << 20)) >> 21; s4 += carry; s3 -= carry << 21;
    carry = (s5 + (1 << 20)) >> 21; s6 += carry; s5 -= carry << 21;
    carry = (s7 + (1 << 20)) >> 21; s8 += carry; s7 -= carry << 21;
    carry = (s9 + (1 << 20)) >> 21; s10 += carry; s9 -= carry << 21;
    carry = (s11 + (1 << 20)) >> 21; s12 += carry; s11 -= carry << 21;

    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;

    // Unsigned carries after first s12 reduction
    carry = s0 >> 21; s1 += carry; s0 -= carry << 21;
    carry = s1 >> 21; s2 += carry; s1 -= carry << 21;
    carry = s2 >> 21; s3 += carry; s2 -= carry << 21;
    carry = s3 >> 21; s4 += carry; s3 -= carry << 21;
    carry = s4 >> 21; s5 += carry; s4 -= carry << 21;
    carry = s5 >> 21; s6 += carry; s5 -= carry << 21;
    carry = s6 >> 21; s7 += carry; s6 -= carry << 21;
    carry = s7 >> 21; s8 += carry; s7 -= carry << 21;
    carry = s8 >> 21; s9 += carry; s8 -= carry << 21;
    carry = s9 >> 21; s10 += carry; s9 -= carry << 21;
    carry = s10 >> 21; s11 += carry; s10 -= carry << 21;
    carry = s11 >> 21; s12 += carry; s11 -= carry << 21;

    // Second s12 reduction
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;

    // Final unsigned carries
    carry = s0 >> 21; s1 += carry; s0 -= carry << 21;
    carry = s1 >> 21; s2 += carry; s1 -= carry << 21;
    carry = s2 >> 21; s3 += carry; s2 -= carry << 21;
    carry = s3 >> 21; s4 += carry; s3 -= carry << 21;
    carry = s4 >> 21; s5 += carry; s4 -= carry << 21;
    carry = s5 >> 21; s6 += carry; s5 -= carry << 21;
    carry = s6 >> 21; s7 += carry; s6 -= carry << 21;
    carry = s7 >> 21; s8 += carry; s7 -= carry << 21;
    carry = s8 >> 21; s9 += carry; s8 -= carry << 21;
    carry = s9 >> 21; s10 += carry; s9 -= carry << 21;
    carry = s10 >> 21; s11 += carry; s10 -= carry << 21;

    // Pack result into 32 bytes
    s[0] = static_cast<uint8_t>(s0);
    s[1] = static_cast<uint8_t>(s0 >> 8);
    s[2] = static_cast<uint8_t>((s0 >> 16) | (s1 << 5));
    s[3] = static_cast<uint8_t>(s1 >> 3);
    s[4] = static_cast<uint8_t>(s1 >> 11);
    s[5] = static_cast<uint8_t>((s1 >> 19) | (s2 << 2));
    s[6] = static_cast<uint8_t>(s2 >> 6);
    s[7] = static_cast<uint8_t>((s2 >> 14) | (s3 << 7));
    s[8] = static_cast<uint8_t>(s3 >> 1);
    s[9] = static_cast<uint8_t>(s3 >> 9);
    s[10] = static_cast<uint8_t>((s3 >> 17) | (s4 << 4));
    s[11] = static_cast<uint8_t>(s4 >> 4);
    s[12] = static_cast<uint8_t>(s4 >> 12);
    s[13] = static_cast<uint8_t>((s4 >> 20) | (s5 << 1));
    s[14] = static_cast<uint8_t>(s5 >> 7);
    s[15] = static_cast<uint8_t>((s5 >> 15) | (s6 << 6));
    s[16] = static_cast<uint8_t>(s6 >> 2);
    s[17] = static_cast<uint8_t>(s6 >> 10);
    s[18] = static_cast<uint8_t>((s6 >> 18) | (s7 << 3));
    s[19] = static_cast<uint8_t>(s7 >> 5);
    s[20] = static_cast<uint8_t>(s7 >> 13);
    s[21] = static_cast<uint8_t>(s8);
    s[22] = static_cast<uint8_t>(s8 >> 8);
    s[23] = static_cast<uint8_t>((s8 >> 16) | (s9 << 5));
    s[24] = static_cast<uint8_t>(s9 >> 3);
    s[25] = static_cast<uint8_t>(s9 >> 11);
    s[26] = static_cast<uint8_t>((s9 >> 19) | (s10 << 2));
    s[27] = static_cast<uint8_t>(s10 >> 6);
    s[28] = static_cast<uint8_t>((s10 >> 14) | (s11 << 7));
    s[29] = static_cast<uint8_t>(s11 >> 1);
    s[30] = static_cast<uint8_t>(s11 >> 9);
    s[31] = static_cast<uint8_t>(s11 >> 17);
}

// Compute s = (a * b + c) mod L
// All inputs are 32 bytes (256 bits), little-endian
// Result is stored in s (32 bytes)
inline void sc25519_muladd(uint8_t* s, const uint8_t* a, const uint8_t* b, const uint8_t* c) {
    // Load a, b, c as 21-bit limbs (12 limbs each for 32-byte inputs)
    // Uses load3/load4 to avoid reading beyond the input buffers
    int64_t a0  = 0x1fffff & sc25519_load3(a);
    int64_t a1  = 0x1fffff & (sc25519_load4(a + 2) >> 5);
    int64_t a2  = 0x1fffff & (sc25519_load3(a + 5) >> 2);
    int64_t a3  = 0x1fffff & (sc25519_load4(a + 7) >> 7);
    int64_t a4  = 0x1fffff & (sc25519_load4(a + 10) >> 4);
    int64_t a5  = 0x1fffff & (sc25519_load3(a + 13) >> 1);
    int64_t a6  = 0x1fffff & (sc25519_load4(a + 15) >> 6);
    int64_t a7  = 0x1fffff & (sc25519_load3(a + 18) >> 3);
    int64_t a8  = 0x1fffff & sc25519_load3(a + 21);
    int64_t a9  = 0x1fffff & (sc25519_load4(a + 23) >> 5);
    int64_t a10 = 0x1fffff & (sc25519_load3(a + 26) >> 2);
    int64_t a11 = (sc25519_load4(a + 28) >> 7);

    int64_t b0  = 0x1fffff & sc25519_load3(b);
    int64_t b1  = 0x1fffff & (sc25519_load4(b + 2) >> 5);
    int64_t b2  = 0x1fffff & (sc25519_load3(b + 5) >> 2);
    int64_t b3  = 0x1fffff & (sc25519_load4(b + 7) >> 7);
    int64_t b4  = 0x1fffff & (sc25519_load4(b + 10) >> 4);
    int64_t b5  = 0x1fffff & (sc25519_load3(b + 13) >> 1);
    int64_t b6  = 0x1fffff & (sc25519_load4(b + 15) >> 6);
    int64_t b7  = 0x1fffff & (sc25519_load3(b + 18) >> 3);
    int64_t b8  = 0x1fffff & sc25519_load3(b + 21);
    int64_t b9  = 0x1fffff & (sc25519_load4(b + 23) >> 5);
    int64_t b10 = 0x1fffff & (sc25519_load3(b + 26) >> 2);
    int64_t b11 = (sc25519_load4(b + 28) >> 7);

    int64_t c0  = 0x1fffff & sc25519_load3(c);
    int64_t c1  = 0x1fffff & (sc25519_load4(c + 2) >> 5);
    int64_t c2  = 0x1fffff & (sc25519_load3(c + 5) >> 2);
    int64_t c3  = 0x1fffff & (sc25519_load4(c + 7) >> 7);
    int64_t c4  = 0x1fffff & (sc25519_load4(c + 10) >> 4);
    int64_t c5  = 0x1fffff & (sc25519_load3(c + 13) >> 1);
    int64_t c6  = 0x1fffff & (sc25519_load4(c + 15) >> 6);
    int64_t c7  = 0x1fffff & (sc25519_load3(c + 18) >> 3);
    int64_t c8  = 0x1fffff & sc25519_load3(c + 21);
    int64_t c9  = 0x1fffff & (sc25519_load4(c + 23) >> 5);
    int64_t c10 = 0x1fffff & (sc25519_load3(c + 26) >> 2);
    int64_t c11 = (sc25519_load4(c + 28) >> 7);

    // Compute s = a*b + c (schoolbook multiplication)
    int64_t s0  = c0  + a0*b0;
    int64_t s1  = c1  + a0*b1  + a1*b0;
    int64_t s2  = c2  + a0*b2  + a1*b1  + a2*b0;
    int64_t s3  = c3  + a0*b3  + a1*b2  + a2*b1  + a3*b0;
    int64_t s4  = c4  + a0*b4  + a1*b3  + a2*b2  + a3*b1  + a4*b0;
    int64_t s5  = c5  + a0*b5  + a1*b4  + a2*b3  + a3*b2  + a4*b1  + a5*b0;
    int64_t s6  = c6  + a0*b6  + a1*b5  + a2*b4  + a3*b3  + a4*b2  + a5*b1  + a6*b0;
    int64_t s7  = c7  + a0*b7  + a1*b6  + a2*b5  + a3*b4  + a4*b3  + a5*b2  + a6*b1  + a7*b0;
    int64_t s8  = c8  + a0*b8  + a1*b7  + a2*b6  + a3*b5  + a4*b4  + a5*b3  + a6*b2  + a7*b1  + a8*b0;
    int64_t s9  = c9  + a0*b9  + a1*b8  + a2*b7  + a3*b6  + a4*b5  + a5*b4  + a6*b3  + a7*b2  + a8*b1  + a9*b0;
    int64_t s10 = c10 + a0*b10 + a1*b9  + a2*b8  + a3*b7  + a4*b6  + a5*b5  + a6*b4  + a7*b3  + a8*b2  + a9*b1  + a10*b0;
    int64_t s11 = c11 + a0*b11 + a1*b10 + a2*b9  + a3*b8  + a4*b7  + a5*b6  + a6*b5  + a7*b4  + a8*b3  + a9*b2  + a10*b1 + a11*b0;
    int64_t s12 =       a1*b11 + a2*b10 + a3*b9  + a4*b8  + a5*b7  + a6*b6  + a7*b5  + a8*b4  + a9*b3  + a10*b2 + a11*b1;
    int64_t s13 =                a2*b11 + a3*b10 + a4*b9  + a5*b8  + a6*b7  + a7*b6  + a8*b5  + a9*b4  + a10*b3 + a11*b2;
    int64_t s14 =                         a3*b11 + a4*b10 + a5*b9  + a6*b8  + a7*b7  + a8*b6  + a9*b5  + a10*b4 + a11*b3;
    int64_t s15 =                                  a4*b11 + a5*b10 + a6*b9  + a7*b8  + a8*b7  + a9*b6  + a10*b5 + a11*b4;
    int64_t s16 =                                           a5*b11 + a6*b10 + a7*b9  + a8*b8  + a9*b7  + a10*b6 + a11*b5;
    int64_t s17 =                                                    a6*b11 + a7*b10 + a8*b9  + a9*b8  + a10*b7 + a11*b6;
    int64_t s18 =                                                             a7*b11 + a8*b10 + a9*b9  + a10*b8 + a11*b7;
    int64_t s19 =                                                                      a8*b11 + a9*b10 + a10*b9 + a11*b8;
    int64_t s20 =                                                                               a9*b11 + a10*b10 + a11*b9;
    int64_t s21 =                                                                                        a10*b11 + a11*b10;
    int64_t s22 =                                                                                                  a11*b11;
    int64_t s23 = 0;

    int64_t carry;

    // Reduce using: 2^252 ≡ -mu (mod L)
    // Same reduction as sc25519_reduce

    carry = (s0 + (1 << 20)) >> 21; s1 += carry; s0 -= carry << 21;
    carry = (s2 + (1 << 20)) >> 21; s3 += carry; s2 -= carry << 21;
    carry = (s4 + (1 << 20)) >> 21; s5 += carry; s4 -= carry << 21;
    carry = (s6 + (1 << 20)) >> 21; s7 += carry; s6 -= carry << 21;
    carry = (s8 + (1 << 20)) >> 21; s9 += carry; s8 -= carry << 21;
    carry = (s10 + (1 << 20)) >> 21; s11 += carry; s10 -= carry << 21;
    carry = (s12 + (1 << 20)) >> 21; s13 += carry; s12 -= carry << 21;
    carry = (s14 + (1 << 20)) >> 21; s15 += carry; s14 -= carry << 21;
    carry = (s16 + (1 << 20)) >> 21; s17 += carry; s16 -= carry << 21;
    carry = (s18 + (1 << 20)) >> 21; s19 += carry; s18 -= carry << 21;
    carry = (s20 + (1 << 20)) >> 21; s21 += carry; s20 -= carry << 21;
    carry = (s22 + (1 << 20)) >> 21; s23 += carry; s22 -= carry << 21;

    carry = (s1 + (1 << 20)) >> 21; s2 += carry; s1 -= carry << 21;
    carry = (s3 + (1 << 20)) >> 21; s4 += carry; s3 -= carry << 21;
    carry = (s5 + (1 << 20)) >> 21; s6 += carry; s5 -= carry << 21;
    carry = (s7 + (1 << 20)) >> 21; s8 += carry; s7 -= carry << 21;
    carry = (s9 + (1 << 20)) >> 21; s10 += carry; s9 -= carry << 21;
    carry = (s11 + (1 << 20)) >> 21; s12 += carry; s11 -= carry << 21;
    carry = (s13 + (1 << 20)) >> 21; s14 += carry; s13 -= carry << 21;
    carry = (s15 + (1 << 20)) >> 21; s16 += carry; s15 -= carry << 21;
    carry = (s17 + (1 << 20)) >> 21; s18 += carry; s17 -= carry << 21;
    carry = (s19 + (1 << 20)) >> 21; s20 += carry; s19 -= carry << 21;
    carry = (s21 + (1 << 20)) >> 21; s22 += carry; s21 -= carry << 21;

    s11 += s23 * 666643;
    s12 += s23 * 470296;
    s13 += s23 * 654183;
    s14 -= s23 * 997805;
    s15 += s23 * 136657;
    s16 -= s23 * 683901;
    s23 = 0;

    s10 += s22 * 666643;
    s11 += s22 * 470296;
    s12 += s22 * 654183;
    s13 -= s22 * 997805;
    s14 += s22 * 136657;
    s15 -= s22 * 683901;
    s22 = 0;

    s9 += s21 * 666643;
    s10 += s21 * 470296;
    s11 += s21 * 654183;
    s12 -= s21 * 997805;
    s13 += s21 * 136657;
    s14 -= s21 * 683901;
    s21 = 0;

    s8 += s20 * 666643;
    s9 += s20 * 470296;
    s10 += s20 * 654183;
    s11 -= s20 * 997805;
    s12 += s20 * 136657;
    s13 -= s20 * 683901;
    s20 = 0;

    s7 += s19 * 666643;
    s8 += s19 * 470296;
    s9 += s19 * 654183;
    s10 -= s19 * 997805;
    s11 += s19 * 136657;
    s12 -= s19 * 683901;
    s19 = 0;

    s6 += s18 * 666643;
    s7 += s18 * 470296;
    s8 += s18 * 654183;
    s9 -= s18 * 997805;
    s10 += s18 * 136657;
    s11 -= s18 * 683901;
    s18 = 0;

    carry = (s6 + (1 << 20)) >> 21; s7 += carry; s6 -= carry << 21;
    carry = (s8 + (1 << 20)) >> 21; s9 += carry; s8 -= carry << 21;
    carry = (s10 + (1 << 20)) >> 21; s11 += carry; s10 -= carry << 21;
    carry = (s12 + (1 << 20)) >> 21; s13 += carry; s12 -= carry << 21;
    carry = (s14 + (1 << 20)) >> 21; s15 += carry; s14 -= carry << 21;
    carry = (s16 + (1 << 20)) >> 21; s17 += carry; s16 -= carry << 21;

    carry = (s7 + (1 << 20)) >> 21; s8 += carry; s7 -= carry << 21;
    carry = (s9 + (1 << 20)) >> 21; s10 += carry; s9 -= carry << 21;
    carry = (s11 + (1 << 20)) >> 21; s12 += carry; s11 -= carry << 21;
    carry = (s13 + (1 << 20)) >> 21; s14 += carry; s13 -= carry << 21;
    carry = (s15 + (1 << 20)) >> 21; s16 += carry; s15 -= carry << 21;

    s5 += s17 * 666643;
    s6 += s17 * 470296;
    s7 += s17 * 654183;
    s8 -= s17 * 997805;
    s9 += s17 * 136657;
    s10 -= s17 * 683901;
    s17 = 0;

    s4 += s16 * 666643;
    s5 += s16 * 470296;
    s6 += s16 * 654183;
    s7 -= s16 * 997805;
    s8 += s16 * 136657;
    s9 -= s16 * 683901;
    s16 = 0;

    s3 += s15 * 666643;
    s4 += s15 * 470296;
    s5 += s15 * 654183;
    s6 -= s15 * 997805;
    s7 += s15 * 136657;
    s8 -= s15 * 683901;
    s15 = 0;

    s2 += s14 * 666643;
    s3 += s14 * 470296;
    s4 += s14 * 654183;
    s5 -= s14 * 997805;
    s6 += s14 * 136657;
    s7 -= s14 * 683901;
    s14 = 0;

    s1 += s13 * 666643;
    s2 += s13 * 470296;
    s3 += s13 * 654183;
    s4 -= s13 * 997805;
    s5 += s13 * 136657;
    s6 -= s13 * 683901;
    s13 = 0;

    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;

    carry = (s0 + (1 << 20)) >> 21; s1 += carry; s0 -= carry << 21;
    carry = (s2 + (1 << 20)) >> 21; s3 += carry; s2 -= carry << 21;
    carry = (s4 + (1 << 20)) >> 21; s5 += carry; s4 -= carry << 21;
    carry = (s6 + (1 << 20)) >> 21; s7 += carry; s6 -= carry << 21;
    carry = (s8 + (1 << 20)) >> 21; s9 += carry; s8 -= carry << 21;
    carry = (s10 + (1 << 20)) >> 21; s11 += carry; s10 -= carry << 21;

    carry = (s1 + (1 << 20)) >> 21; s2 += carry; s1 -= carry << 21;
    carry = (s3 + (1 << 20)) >> 21; s4 += carry; s3 -= carry << 21;
    carry = (s5 + (1 << 20)) >> 21; s6 += carry; s5 -= carry << 21;
    carry = (s7 + (1 << 20)) >> 21; s8 += carry; s7 -= carry << 21;
    carry = (s9 + (1 << 20)) >> 21; s10 += carry; s9 -= carry << 21;
    carry = (s11 + (1 << 20)) >> 21; s12 += carry; s11 -= carry << 21;

    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;

    carry = (s0 + (1 << 20)) >> 21; s1 += carry; s0 -= carry << 21;
    carry = (s1 + (1 << 20)) >> 21; s2 += carry; s1 -= carry << 21;
    carry = (s2 + (1 << 20)) >> 21; s3 += carry; s2 -= carry << 21;
    carry = (s3 + (1 << 20)) >> 21; s4 += carry; s3 -= carry << 21;
    carry = (s4 + (1 << 20)) >> 21; s5 += carry; s4 -= carry << 21;
    carry = (s5 + (1 << 20)) >> 21; s6 += carry; s5 -= carry << 21;
    carry = (s6 + (1 << 20)) >> 21; s7 += carry; s6 -= carry << 21;
    carry = (s7 + (1 << 20)) >> 21; s8 += carry; s7 -= carry << 21;
    carry = (s8 + (1 << 20)) >> 21; s9 += carry; s8 -= carry << 21;
    carry = (s9 + (1 << 20)) >> 21; s10 += carry; s9 -= carry << 21;
    carry = (s10 + (1 << 20)) >> 21; s11 += carry; s10 -= carry << 21;
    carry = (s11 + (1 << 20)) >> 21; s12 += carry; s11 -= carry << 21;

    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;

    carry = s0 >> 21; s1 += carry; s0 -= carry << 21;
    carry = s1 >> 21; s2 += carry; s1 -= carry << 21;
    carry = s2 >> 21; s3 += carry; s2 -= carry << 21;
    carry = s3 >> 21; s4 += carry; s3 -= carry << 21;
    carry = s4 >> 21; s5 += carry; s4 -= carry << 21;
    carry = s5 >> 21; s6 += carry; s5 -= carry << 21;
    carry = s6 >> 21; s7 += carry; s6 -= carry << 21;
    carry = s7 >> 21; s8 += carry; s7 -= carry << 21;
    carry = s8 >> 21; s9 += carry; s8 -= carry << 21;
    carry = s9 >> 21; s10 += carry; s9 -= carry << 21;
    carry = s10 >> 21; s11 += carry; s10 -= carry << 21;
    carry = s11 >> 21; s12 += carry; s11 -= carry << 21;

    // Third s12 reduction
    s0 += s12 * 666643;
    s1 += s12 * 470296;
    s2 += s12 * 654183;
    s3 -= s12 * 997805;
    s4 += s12 * 136657;
    s5 -= s12 * 683901;
    s12 = 0;

    // Final carry propagation
    carry = s0 >> 21; s1 += carry; s0 -= carry << 21;
    carry = s1 >> 21; s2 += carry; s1 -= carry << 21;
    carry = s2 >> 21; s3 += carry; s2 -= carry << 21;
    carry = s3 >> 21; s4 += carry; s3 -= carry << 21;
    carry = s4 >> 21; s5 += carry; s4 -= carry << 21;
    carry = s5 >> 21; s6 += carry; s5 -= carry << 21;
    carry = s6 >> 21; s7 += carry; s6 -= carry << 21;
    carry = s7 >> 21; s8 += carry; s7 -= carry << 21;
    carry = s8 >> 21; s9 += carry; s8 -= carry << 21;
    carry = s9 >> 21; s10 += carry; s9 -= carry << 21;
    carry = s10 >> 21; s11 += carry; s10 -= carry << 21;

    // Pack result
    s[0] = static_cast<uint8_t>(s0);
    s[1] = static_cast<uint8_t>(s0 >> 8);
    s[2] = static_cast<uint8_t>((s0 >> 16) | (s1 << 5));
    s[3] = static_cast<uint8_t>(s1 >> 3);
    s[4] = static_cast<uint8_t>(s1 >> 11);
    s[5] = static_cast<uint8_t>((s1 >> 19) | (s2 << 2));
    s[6] = static_cast<uint8_t>(s2 >> 6);
    s[7] = static_cast<uint8_t>((s2 >> 14) | (s3 << 7));
    s[8] = static_cast<uint8_t>(s3 >> 1);
    s[9] = static_cast<uint8_t>(s3 >> 9);
    s[10] = static_cast<uint8_t>((s3 >> 17) | (s4 << 4));
    s[11] = static_cast<uint8_t>(s4 >> 4);
    s[12] = static_cast<uint8_t>(s4 >> 12);
    s[13] = static_cast<uint8_t>((s4 >> 20) | (s5 << 1));
    s[14] = static_cast<uint8_t>(s5 >> 7);
    s[15] = static_cast<uint8_t>((s5 >> 15) | (s6 << 6));
    s[16] = static_cast<uint8_t>(s6 >> 2);
    s[17] = static_cast<uint8_t>(s6 >> 10);
    s[18] = static_cast<uint8_t>((s6 >> 18) | (s7 << 3));
    s[19] = static_cast<uint8_t>(s7 >> 5);
    s[20] = static_cast<uint8_t>(s7 >> 13);
    s[21] = static_cast<uint8_t>(s8);
    s[22] = static_cast<uint8_t>(s8 >> 8);
    s[23] = static_cast<uint8_t>((s8 >> 16) | (s9 << 5));
    s[24] = static_cast<uint8_t>(s9 >> 3);
    s[25] = static_cast<uint8_t>(s9 >> 11);
    s[26] = static_cast<uint8_t>((s9 >> 19) | (s10 << 2));
    s[27] = static_cast<uint8_t>(s10 >> 6);
    s[28] = static_cast<uint8_t>((s10 >> 14) | (s11 << 7));
    s[29] = static_cast<uint8_t>(s11 >> 1);
    s[30] = static_cast<uint8_t>(s11 >> 9);
    s[31] = static_cast<uint8_t>(s11 >> 17);
}

// Check if a scalar is less than L (the curve order)
// Returns true if s < L
inline bool sc25519_is_canonical(const uint8_t* s) {
    // L in little-endian:
    // ed d3 f5 5c 1a 63 12 58 d6 9c f7 a2 de f9 de 14
    // 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10
    static const uint8_t L[32] = {
        0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
        0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10
    };

    // Compare from high byte to low byte
    for (int i = 31; i >= 0; i--) {
        if (s[i] < L[i]) return true;
        if (s[i] > L[i]) return false;
    }
    // s == L, which is not canonical (should be reduced to 0)
    return false;
}

// ============================================================================
// Ed25519 Elliptic Curve Point Operations
// ============================================================================
//
// Ed25519 uses the twisted Edwards curve: -x² + y² = 1 + d*x²*y²
// where d = -121665/121666 (mod p), p = 2^255 - 19
//
// Points are represented in extended coordinates (X, Y, Z, T) where:
//   x = X/Z, y = Y/Z, and T = XY/Z
//
// This representation enables efficient point addition with unified formulas.

// ============================================================================
// Curve Constants
// ============================================================================

// d = -121665/121666 (mod p)
// Computed as: -121665 * inverse(121666) mod p
// In radix 2^51 representation:
inline const Fe25519& ge25519_d() {
    static const Fe25519 d(
        929955233495203LL,
        466365720129213LL,
        1662059464998953LL,
        2033849074728123LL,
        1442794654840575LL
    );
    return d;
}

// 2*d = 2 * (-121665/121666) (mod p)
// Used in point addition formula
inline const Fe25519& ge25519_2d() {
    static const Fe25519 d2(
        1859910466990425LL,
        932731440258426LL,
        1072319116312658LL,
        1815898335770999LL,
        633789495995903LL
    );
    return d2;
}

// sqrt(-1) mod p = 2^((p-1)/4) mod p
// Used for computing square roots
inline const Fe25519& fe25519_sqrtm1() {
    static const Fe25519 sqrtm1(
        1718705420411056LL,
        234908883556509LL,
        2233514472574048LL,
        2117202627021982LL,
        765476049583133LL
    );
    return sqrtm1;
}

// ============================================================================
// Ed25519 Point in Extended Coordinates
// ============================================================================

// Point on Ed25519 curve in extended coordinates
// Represents point (x, y) where x = X/Z, y = Y/Z, T = XY/Z
struct Ge25519 {
    Fe25519 X, Y, Z, T;
};

// ============================================================================
// Point Operations
// ============================================================================

// Set point to identity (0, 1, 1, 0) in extended coordinates
// Identity point on twisted Edwards curve is (0, 1)
inline void ge25519_zero(Ge25519& h) {
    fe25519_zero(h.X);
    fe25519_one(h.Y);
    fe25519_one(h.Z);
    fe25519_zero(h.T);
}

// Copy point: r = p
inline void ge25519_copy(Ge25519& r, const Ge25519& p) {
    fe25519_copy(r.X, p.X);
    fe25519_copy(r.Y, p.Y);
    fe25519_copy(r.Z, p.Z);
    fe25519_copy(r.T, p.T);
}

// Point addition using unified extended coordinate formulas
// r = p + q
//
// Algorithm (RFC 8032, Section 5.1.4):
//   A = (Y1 - X1) * (Y2 - X2)
//   B = (Y1 + X1) * (Y2 + X2)
//   C = T1 * 2*d * T2
//   D = Z1 * 2 * Z2
//   E = B - A
//   F = D - C
//   G = D + C
//   H = B + A
//   X3 = E * F
//   Y3 = G * H
//   T3 = E * H
//   Z3 = F * G
inline void ge25519_add(Ge25519& r, const Ge25519& p, const Ge25519& q) {
    Fe25519 A, B, C, D, E, F, G, H, t0, t1;

    // A = (Y1 - X1) * (Y2 - X2)
    fe25519_sub(t0, p.Y, p.X);
    fe25519_sub(t1, q.Y, q.X);
    fe25519_mul(A, t0, t1);

    // B = (Y1 + X1) * (Y2 + X2)
    fe25519_add(t0, p.Y, p.X);
    fe25519_add(t1, q.Y, q.X);
    fe25519_mul(B, t0, t1);

    // C = T1 * 2*d * T2
    fe25519_mul(t0, p.T, q.T);
    fe25519_mul(C, t0, ge25519_2d());

    // D = Z1 * 2 * Z2
    fe25519_mul(t0, p.Z, q.Z);
    fe25519_add(D, t0, t0);  // D = 2 * Z1 * Z2

    // E = B - A
    fe25519_sub(E, B, A);

    // F = D - C
    fe25519_sub(F, D, C);

    // G = D + C
    fe25519_add(G, D, C);

    // H = B + A
    fe25519_add(H, B, A);

    // X3 = E * F
    fe25519_mul(r.X, E, F);

    // Y3 = G * H
    fe25519_mul(r.Y, G, H);

    // T3 = E * H
    fe25519_mul(r.T, E, H);

    // Z3 = F * G
    fe25519_mul(r.Z, F, G);
}

// Point doubling using optimized formula
// r = 2 * p
//
// Algorithm (RFC 8032, optimized doubling):
//   A = X1^2
//   B = Y1^2
//   C = 2 * Z1^2
//   H = A + B
//   E = H - (X1 + Y1)^2
//   G = A - B
//   F = C + G
//   X3 = E * F
//   Y3 = G * H
//   T3 = E * H
//   Z3 = F * G
inline void ge25519_double(Ge25519& r, const Ge25519& p) {
    Fe25519 A, B, C, E, F, G, H, t0;

    // A = X1^2
    fe25519_sq(A, p.X);

    // B = Y1^2
    fe25519_sq(B, p.Y);

    // C = 2 * Z1^2
    fe25519_sq(t0, p.Z);
    fe25519_add(C, t0, t0);

    // H = A + B
    fe25519_add(H, A, B);

    // E = H - (X1 + Y1)^2
    fe25519_add(t0, p.X, p.Y);
    fe25519_sq(t0, t0);
    fe25519_sub(E, H, t0);

    // G = A - B
    fe25519_sub(G, A, B);

    // F = C + G
    fe25519_add(F, C, G);

    // X3 = E * F
    fe25519_mul(r.X, E, F);

    // Y3 = G * H
    fe25519_mul(r.Y, G, H);

    // T3 = E * H
    fe25519_mul(r.T, E, H);

    // Z3 = F * G
    fe25519_mul(r.Z, F, G);
}

// Scalar multiplication using double-and-add algorithm
// r = s * p where s is a 256-bit scalar
//
// Note: This is a simple implementation. For production use, consider
// constant-time implementations to prevent timing attacks.
inline void ge25519_scalarmult(Ge25519& r, const uint8_t* s, const Ge25519& p) {
    Ge25519 Q;
    ge25519_zero(Q);  // Q = identity

    // Process bits from most significant to least significant
    for (int i = 255; i >= 0; i--) {
        // Double
        ge25519_double(Q, Q);

        // Add if bit is set
        int byte_idx = i >> 3;
        int bit_idx = i & 7;
        int bit = (s[byte_idx] >> bit_idx) & 1;

        if (bit) {
            ge25519_add(Q, Q, p);
        }
    }

    ge25519_copy(r, Q);
}

// Constant-time scalar multiplication
// r = s * p where s is a 256-bit scalar
// Uses constant-time conditional moves for side-channel resistance
inline void ge25519_scalarmult_ct(Ge25519& r, const uint8_t* s, const Ge25519& p) {
    Ge25519 Q, T;
    ge25519_zero(Q);  // Q = identity

    // Process bits from most significant to least significant
    for (int i = 255; i >= 0; i--) {
        // Double
        ge25519_double(Q, Q);

        // Compute Q + p
        ge25519_add(T, Q, p);

        // Extract bit
        int byte_idx = i >> 3;
        int bit_idx = i & 7;
        int64_t bit = (s[byte_idx] >> bit_idx) & 1;

        // Conditional move: Q = bit ? T : Q
        fe25519_cmov(Q.X, T.X, bit);
        fe25519_cmov(Q.Y, T.Y, bit);
        fe25519_cmov(Q.Z, T.Z, bit);
        fe25519_cmov(Q.T, T.T, bit);
    }

    ge25519_copy(r, Q);
}

// ============================================================================
// Base Point
// ============================================================================

// Ed25519 base point G with y = 4/5 (mod p)
// x is the positive square root of (y^2 - 1) / (d*y^2 + 1)
//
// Base point coordinates:
// y = 4/5 mod p = 4 * inverse(5) mod p
// x = sqrt((y^2 - 1) / (d*y^2 + 1)) with x positive (even)
inline const Ge25519& ge25519_basepoint() {
    // Base point X coordinate in radix 2^51
    static const Fe25519 base_X(
        1738742601995546LL,
        1146398526822698LL,
        2070867633025821LL,
        562264141797630LL,
        587772402128613LL
    );

    // Base point Y coordinate = 4/5 mod p in radix 2^51
    static const Fe25519 base_Y(
        1801439850948184LL,
        1351079888211148LL,
        450359962737049LL,
        900719925474099LL,
        1801439850948198LL
    );

    static Ge25519 G;
    static bool initialized = false;

    if (!initialized) {
        fe25519_copy(G.X, base_X);
        fe25519_copy(G.Y, base_Y);
        fe25519_one(G.Z);
        fe25519_mul(G.T, G.X, G.Y);  // T = X * Y
        initialized = true;
    }

    return G;
}

// Base point scalar multiplication
// r = s * G where G is the standard Ed25519 base point
inline void ge25519_scalarmult_base(Ge25519& r, const uint8_t* s) {
    ge25519_scalarmult_ct(r, s, ge25519_basepoint());
}

// ============================================================================
// Point Encoding/Decoding (RFC 8032)
// ============================================================================

// Encode point to 32 bytes
// Compression: store y-coordinate with sign bit of x in the top bit (bit 255)
//
// Algorithm:
//   1. Compute affine coordinates: x = X/Z, y = Y/Z
//   2. Store y as 32 bytes (little-endian)
//   3. Set bit 255 to the least significant bit of x
inline void ge25519_tobytes(uint8_t* s, const Ge25519& h) {
    Fe25519 x, y, z_inv;

    // Compute z_inv = 1/Z
    fe25519_invert(z_inv, h.Z);

    // Compute affine coordinates
    fe25519_mul(x, h.X, z_inv);  // x = X/Z
    fe25519_mul(y, h.Y, z_inv);  // y = Y/Z

    // Store y as 32 bytes
    fe25519_tobytes(s, y);

    // Set bit 255 to sign of x (LSB of x after reduction)
    s[31] ^= static_cast<uint8_t>(fe25519_isnegative(x) << 7);
}

// Compute square root of u/v, returning success status
// If u/v is a square, sets r to sqrt(u/v) and returns true
// Otherwise returns false
//
// Uses the identity: sqrt(u/v) = u * v^3 * (u * v^7)^((p-5)/8)
// For p = 2^255 - 19, (p-5)/8 = 2^252 - 3
inline bool fe25519_sqrt_ratio(Fe25519& r, const Fe25519& u, const Fe25519& v) {
    Fe25519 v3, v7, uv7, x, check;

    // v3 = v^3
    fe25519_sq(v3, v);       // v^2
    fe25519_mul(v3, v3, v);  // v^3

    // v7 = v^7
    fe25519_sq(v7, v3);      // v^6
    fe25519_mul(v7, v7, v);  // v^7

    // uv7 = u * v^7
    fe25519_mul(uv7, u, v7);

    // x = (u * v^7)^((p-5)/8)
    fe25519_pow2523(x, uv7);

    // r = u * v^3 * x
    fe25519_mul(r, u, v3);
    fe25519_mul(r, r, x);

    // check = v * r^2
    fe25519_sq(check, r);
    fe25519_mul(check, check, v);

    // Verify: check should equal u or -u
    Fe25519 neg_u;
    fe25519_neg(neg_u, u);

    // If check == u, we're done
    // If check == -u, multiply r by sqrt(-1)
    // Otherwise, no square root exists

    // Compute check - u and check + u (which is check - (-u))
    Fe25519 diff1, diff2;
    fe25519_sub(diff1, check, u);
    fe25519_sub(diff2, check, neg_u);

    int correct = fe25519_iszero(diff1);
    int flipped = fe25519_iszero(diff2);

    // If flipped, multiply r by sqrt(-1)
    Fe25519 r_prime;
    fe25519_mul(r_prime, r, fe25519_sqrtm1());
    fe25519_cmov(r, r_prime, flipped);

    return (correct || flipped);
}

// Decode 32 bytes to point
// Returns false if the encoding is invalid
//
// Algorithm:
//   1. Extract y from bytes (clear bit 255)
//   2. Extract sign bit x_sign from bit 255
//   3. Compute x^2 = (y^2 - 1) / (d*y^2 + 1)
//   4. Compute x = sqrt(x^2)
//   5. If x_sign != x mod 2, negate x
//   6. Verify point is on curve
inline bool ge25519_frombytes(Ge25519& h, const uint8_t* s) {
    Fe25519 u, v, y, x;

    // Extract sign bit
    int x_sign = (s[31] >> 7) & 1;

    // Load y, clearing bit 255
    uint8_t y_bytes[32];
    for (int i = 0; i < 32; i++) {
        y_bytes[i] = s[i];
    }
    y_bytes[31] &= 0x7F;  // Clear bit 255

    fe25519_frombytes(y, y_bytes);

    // Check y < p (already ensured by clearing bit 255 and fe25519_frombytes)

    // Compute u = y^2 - 1
    fe25519_sq(u, y);
    Fe25519 one;
    fe25519_one(one);
    fe25519_sub(u, u, one);

    // Compute v = d*y^2 + 1
    fe25519_sq(v, y);
    fe25519_mul(v, v, ge25519_d());
    fe25519_add(v, v, one);

    // Compute x = sqrt(u/v)
    if (!fe25519_sqrt_ratio(x, u, v)) {
        return false;  // Not a valid point
    }

    // Adjust sign of x
    if (fe25519_isnegative(x) != x_sign) {
        fe25519_neg(x, x);
    }

    // Special case: if x == 0 and x_sign == 1, reject
    if (fe25519_iszero(x) && x_sign) {
        return false;
    }

    // Set point coordinates
    fe25519_copy(h.X, x);
    fe25519_copy(h.Y, y);
    fe25519_one(h.Z);
    fe25519_mul(h.T, x, y);

    return true;
}

// ============================================================================
// Point Validation
// ============================================================================

// Check if point is on the Ed25519 curve
// Verifies: -x^2 + y^2 = 1 + d*x^2*y^2
inline bool ge25519_is_on_curve(const Ge25519& p) {
    Fe25519 x2, y2, lhs, rhs, d_x2_y2, one, z2, z4;

    // Convert to affine coordinates by dividing by Z
    Fe25519 z_inv, x, y;
    fe25519_invert(z_inv, p.Z);
    fe25519_mul(x, p.X, z_inv);
    fe25519_mul(y, p.Y, z_inv);

    // Compute x^2 and y^2
    fe25519_sq(x2, x);
    fe25519_sq(y2, y);

    // lhs = -x^2 + y^2
    fe25519_neg(lhs, x2);
    fe25519_add(lhs, lhs, y2);

    // rhs = 1 + d*x^2*y^2
    fe25519_mul(d_x2_y2, x2, y2);
    fe25519_mul(d_x2_y2, d_x2_y2, ge25519_d());
    fe25519_one(one);
    fe25519_add(rhs, one, d_x2_y2);

    // Check lhs == rhs
    Fe25519 diff;
    fe25519_sub(diff, lhs, rhs);
    return fe25519_iszero(diff) != 0;
}

// Check if point is the identity (neutral element)
inline bool ge25519_is_identity(const Ge25519& p) {
    // Identity in extended coordinates: X = 0, Y = Z
    // (which corresponds to affine (0, 1))
    Fe25519 diff;
    fe25519_sub(diff, p.Y, p.Z);
    return fe25519_iszero(p.X) && fe25519_iszero(diff);
}

// ============================================================================
// Point Negation
// ============================================================================

// Negate a point: r = -p
// On twisted Edwards curve, negation is: (x, y) -> (-x, y)
inline void ge25519_neg(Ge25519& r, const Ge25519& p) {
    fe25519_neg(r.X, p.X);
    fe25519_copy(r.Y, p.Y);
    fe25519_copy(r.Z, p.Z);
    fe25519_neg(r.T, p.T);
}

// ============================================================================
// Point Subtraction
// ============================================================================

// Point subtraction: r = p - q
inline void ge25519_sub(Ge25519& r, const Ge25519& p, const Ge25519& q) {
    Ge25519 neg_q;
    ge25519_neg(neg_q, q);
    ge25519_add(r, p, neg_q);
}

// ============================================================================
// Double Scalar Multiplication
// ============================================================================

// Compute r = s * p + t * G (Straus/Shamir's trick)
// Useful for signature verification
inline void ge25519_double_scalarmult(Ge25519& r, const uint8_t* s,
                                       const Ge25519& p, const uint8_t* t) {
    Ge25519 Q, T1, T2;
    const Ge25519& G = ge25519_basepoint();

    ge25519_zero(Q);  // Q = identity

    // Precompute p + G
    Ge25519 p_plus_G;
    ge25519_add(p_plus_G, p, G);

    // Process bits from most significant to least significant
    for (int i = 255; i >= 0; i--) {
        // Double
        ge25519_double(Q, Q);

        // Extract bits
        int byte_idx = i >> 3;
        int bit_idx = i & 7;
        int s_bit = (s[byte_idx] >> bit_idx) & 1;
        int t_bit = (t[byte_idx] >> bit_idx) & 1;

        // Add based on bit combination
        if (s_bit && t_bit) {
            ge25519_add(Q, Q, p_plus_G);
        } else if (s_bit) {
            ge25519_add(Q, Q, p);
        } else if (t_bit) {
            ge25519_add(Q, Q, G);
        }
    }

    ge25519_copy(r, Q);
}

// ============================================================================
// Point Equality
// ============================================================================

// Check if two points are equal
// Points (X1:Y1:Z1:T1) and (X2:Y2:Z2:T2) are equal iff
// X1*Z2 == X2*Z1 and Y1*Z2 == Y2*Z1
inline bool ge25519_equal(const Ge25519& p, const Ge25519& q) {
    Fe25519 lhs, rhs, diff;

    // Check X1*Z2 == X2*Z1
    fe25519_mul(lhs, p.X, q.Z);
    fe25519_mul(rhs, q.X, p.Z);
    fe25519_sub(diff, lhs, rhs);
    if (!fe25519_iszero(diff)) {
        return false;
    }

    // Check Y1*Z2 == Y2*Z1
    fe25519_mul(lhs, p.Y, q.Z);
    fe25519_mul(rhs, q.Y, p.Z);
    fe25519_sub(diff, lhs, rhs);
    return fe25519_iszero(diff) != 0;
}

// ============================================================================
// Ed25519 Signature Functions (RFC 8032)
// ============================================================================

// Ed25519 scalar clamping (for signing keys)
// Clamps the first 32 bytes of the SHA-512 hash of the seed
inline void ed25519_clamp_scalar(uint8_t* s) {
    s[0] &= 248;   // Clear bits 0, 1, 2
    s[31] &= 127;  // Clear bit 255
    s[31] |= 64;   // Set bit 254
}

// Create Ed25519 keypair from 32-byte seed
//
// Algorithm (RFC 8032, Section 5.1.5):
//   1. h = SHA-512(seed) -> 64 bytes
//   2. s = clamp(h[0:32]) -> scalar
//   3. prefix = h[32:64] -> used in signing
//   4. A = s * G -> public key point
//   5. public_key = encode(A) -> 32 bytes
//   6. private_key = seed || public_key -> 64 bytes
//
// Input:
//   seed: 32-byte random seed
// Output:
//   public_key: 32-byte public key
//   private_key: 64-byte private key (seed || public_key)
inline void ed25519_create_keypair(
    uint8_t* public_key,   // 32 bytes output
    uint8_t* private_key,  // 64 bytes output
    const uint8_t* seed    // 32 bytes input
) {
    uint8_t h[64];

    // Hash the seed
    sha512_hash(h, seed, 32);

    // Clamp the scalar
    ed25519_clamp_scalar(h);

    // Compute public key: A = s * G
    Ge25519 A;
    ge25519_scalarmult_base(A, h);

    // Encode public key
    ge25519_tobytes(public_key, A);

    // Store private key = seed || public_key
    std::memcpy(private_key, seed, 32);
    std::memcpy(private_key + 32, public_key, 32);
}

// Sign a message with Ed25519
//
// Algorithm (RFC 8032, Section 5.1.6):
//   1. h = SHA-512(seed) -> (s, prefix) where s = clamp(h[0:32])
//   2. r = SHA-512(prefix || message) mod L
//   3. R = r * G
//   4. k = SHA-512(encode(R) || public_key || message) mod L
//   5. S = (r + k * s) mod L
//   6. signature = encode(R) || S
//
// Input:
//   message: message to sign
//   message_len: length of message in bytes
//   private_key: 64-byte private key (seed || public_key)
// Output:
//   signature: 64-byte signature (R || S)
inline void ed25519_sign(
    uint8_t* signature,         // 64 bytes output
    const uint8_t* message,
    int64_t message_len,
    const uint8_t* private_key  // 64 bytes (seed || public_key)
) {
    uint8_t h[64];
    uint8_t r_bytes[64];
    uint8_t hram[64];

    // Extract seed from private key
    const uint8_t* seed = private_key;
    const uint8_t* public_key = private_key + 32;

    // Hash the seed to get scalar and prefix
    sha512_hash(h, seed, 32);

    // Clamp the scalar
    uint8_t s[32];
    std::memcpy(s, h, 32);
    ed25519_clamp_scalar(s);

    // prefix = h[32:64]
    const uint8_t* prefix = h + 32;

    // Compute r = SHA-512(prefix || message) mod L
    Sha512Context ctx;
    sha512_init(ctx);
    sha512_update(ctx, prefix, 32);
    sha512_update(ctx, message, message_len);
    sha512_final(ctx, r_bytes);
    sc25519_reduce(r_bytes);

    // R = r * G
    Ge25519 R;
    ge25519_scalarmult_base(R, r_bytes);

    // Encode R
    ge25519_tobytes(signature, R);

    // Compute k = SHA-512(encode(R) || public_key || message) mod L
    sha512_init(ctx);
    sha512_update(ctx, signature, 32);  // encode(R)
    sha512_update(ctx, public_key, 32);
    sha512_update(ctx, message, message_len);
    sha512_final(ctx, hram);
    sc25519_reduce(hram);

    // Compute S = (r + k * s) mod L
    sc25519_muladd(signature + 32, hram, s, r_bytes);
}

// Verify an Ed25519 signature
//
// Algorithm (RFC 8032, Section 5.1.7):
//   1. Decode R from signature[0:32]
//   2. Decode S from signature[32:64], check S < L
//   3. Decode A from public_key
//   4. k = SHA-512(encode(R) || A || message) mod L
//   5. Verify: [8]S*G == [8]R + [8]k*A (cofactor clearing)
//
// Input:
//   signature: 64-byte signature (R || S)
//   message: message that was signed
//   message_len: length of message in bytes
//   public_key: 32-byte public key
// Output:
//   Returns true if signature is valid, false otherwise
inline bool ed25519_verify(
    const uint8_t* signature,   // 64 bytes
    const uint8_t* message,
    int64_t message_len,
    const uint8_t* public_key   // 32 bytes
) {
    // Check that S < L
    if (!sc25519_is_canonical(signature + 32)) {
        return false;
    }

    // Decode public key A
    Ge25519 A;
    if (!ge25519_frombytes(A, public_key)) {
        return false;  // Invalid public key
    }

    // Decode R from signature
    Ge25519 R;
    if (!ge25519_frombytes(R, signature)) {
        return false;  // Invalid R
    }

    // Compute k = SHA-512(encode(R) || A || message) mod L
    uint8_t hram[64];
    Sha512Context ctx;
    sha512_init(ctx);
    sha512_update(ctx, signature, 32);  // encode(R)
    sha512_update(ctx, public_key, 32);
    sha512_update(ctx, message, message_len);
    sha512_final(ctx, hram);
    sc25519_reduce(hram);

    // Compute check_point = S*G - k*A
    // This is equivalent to checking S*G == R + k*A
    //
    // We use the identity: S*G - k*A should equal R
    // Compute: S*G and k*A, then check if S*G - k*A == R

    // Compute S*G
    Ge25519 SG;
    ge25519_scalarmult_base(SG, signature + 32);

    // Compute k*A
    Ge25519 kA;
    ge25519_scalarmult_ct(kA, hram, A);

    // Compute expected = S*G - k*A
    // This should equal R
    Ge25519 expected;
    ge25519_sub(expected, SG, kA);

    // Encode expected point and compare with R
    uint8_t expected_bytes[32];
    ge25519_tobytes(expected_bytes, expected);

    // Constant-time comparison
    uint8_t diff = 0;
    for (int i = 0; i < 32; i++) {
        diff |= expected_bytes[i] ^ signature[i];
    }

    return diff == 0;
}

// Verify an Ed25519 signature with cofactor check
// This version explicitly performs cofactor clearing for extra security
//
// Check: [8]S*G == [8]R + [8]k*A
inline bool ed25519_verify_cofactored(
    const uint8_t* signature,   // 64 bytes
    const uint8_t* message,
    int64_t message_len,
    const uint8_t* public_key   // 32 bytes
) {
    // Check that S < L
    if (!sc25519_is_canonical(signature + 32)) {
        return false;
    }

    // Decode public key A
    Ge25519 A;
    if (!ge25519_frombytes(A, public_key)) {
        return false;  // Invalid public key
    }

    // Decode R from signature
    Ge25519 R;
    if (!ge25519_frombytes(R, signature)) {
        return false;  // Invalid R
    }

    // Compute k = SHA-512(encode(R) || A || message) mod L
    uint8_t hram[64];
    Sha512Context ctx;
    sha512_init(ctx);
    sha512_update(ctx, signature, 32);  // encode(R)
    sha512_update(ctx, public_key, 32);
    sha512_update(ctx, message, message_len);
    sha512_final(ctx, hram);
    sc25519_reduce(hram);

    // Compute S*G
    Ge25519 SG;
    ge25519_scalarmult_base(SG, signature + 32);

    // Compute k*A
    Ge25519 kA;
    ge25519_scalarmult_ct(kA, hram, A);

    // Compute R + k*A
    Ge25519 RkA;
    ge25519_add(RkA, R, kA);

    // Multiply both sides by cofactor 8 = 2^3
    // [8]S*G
    Ge25519 SG8;
    ge25519_double(SG8, SG);   // 2*SG
    ge25519_double(SG8, SG8);  // 4*SG
    ge25519_double(SG8, SG8);  // 8*SG

    // [8](R + k*A)
    Ge25519 RkA8;
    ge25519_double(RkA8, RkA);   // 2*(R + k*A)
    ge25519_double(RkA8, RkA8);  // 4*(R + k*A)
    ge25519_double(RkA8, RkA8);  // 8*(R + k*A)

    // Check if [8]S*G == [8](R + k*A)
    return ge25519_equal(SG8, RkA8);
}

}  // namespace torchscience::kernel::encryption
