// src/torchscience/csrc/kernel/encryption/curve25519.h
#pragma once

#include <cstdint>
#include <array>

namespace torchscience::kernel::encryption {

// Field element for Curve25519: represents integers mod p = 2^255 - 19
// Uses radix 2^51 representation with 5 limbs (each limb holds up to ~51 bits)
// This representation allows efficient multiplication without overflow in int64_t
struct Fe25519 {
    int64_t v[5];

    Fe25519() : v{0, 0, 0, 0, 0} {}
    Fe25519(int64_t v0, int64_t v1, int64_t v2, int64_t v3, int64_t v4)
        : v{v0, v1, v2, v3, v4} {}
};

// Mask for 51 bits
constexpr int64_t FE25519_MASK51 = (1LL << 51) - 1;

// Load 64-bit little-endian word from bytes
inline uint64_t fe25519_load64_le(const uint8_t* bytes) {
    return static_cast<uint64_t>(bytes[0])
         | (static_cast<uint64_t>(bytes[1]) << 8)
         | (static_cast<uint64_t>(bytes[2]) << 16)
         | (static_cast<uint64_t>(bytes[3]) << 24)
         | (static_cast<uint64_t>(bytes[4]) << 32)
         | (static_cast<uint64_t>(bytes[5]) << 40)
         | (static_cast<uint64_t>(bytes[6]) << 48)
         | (static_cast<uint64_t>(bytes[7]) << 56);
}

// Store 64-bit word as little-endian bytes
inline void fe25519_store64_le(uint8_t* bytes, uint64_t word) {
    bytes[0] = static_cast<uint8_t>(word);
    bytes[1] = static_cast<uint8_t>(word >> 8);
    bytes[2] = static_cast<uint8_t>(word >> 16);
    bytes[3] = static_cast<uint8_t>(word >> 24);
    bytes[4] = static_cast<uint8_t>(word >> 32);
    bytes[5] = static_cast<uint8_t>(word >> 40);
    bytes[6] = static_cast<uint8_t>(word >> 48);
    bytes[7] = static_cast<uint8_t>(word >> 56);
}

// Load field element from 32 bytes (little-endian)
// Input bytes are interpreted as a 256-bit integer, which is reduced mod p
inline void fe25519_frombytes(Fe25519& h, const uint8_t* s) {
    // Load 32 bytes as a 256-bit number in little-endian
    // Split into 5 limbs of 51 bits each (with last limb having fewer bits)
    // Limb boundaries: bits 0-50, 51-101, 102-152, 153-203, 204-254

    uint64_t h0 = fe25519_load64_le(s);       // bytes 0-7
    uint64_t h1 = fe25519_load64_le(s + 6);   // bytes 6-13 (overlapping)
    uint64_t h2 = fe25519_load64_le(s + 12);  // bytes 12-19
    uint64_t h3 = fe25519_load64_le(s + 19);  // bytes 19-26
    uint64_t h4 = fe25519_load64_le(s + 24);  // bytes 24-31

    // Extract 51-bit limbs with proper shifting
    h.v[0] = static_cast<int64_t>(h0 & FE25519_MASK51);
    h.v[1] = static_cast<int64_t>((h1 >> 3) & FE25519_MASK51);   // shift by 51-48=3
    h.v[2] = static_cast<int64_t>((h2 >> 6) & FE25519_MASK51);   // shift by 102-96=6
    h.v[3] = static_cast<int64_t>((h3 >> 1) & FE25519_MASK51);   // shift by 153-152=1
    h.v[4] = static_cast<int64_t>((h4 >> 12) & FE25519_MASK51);  // shift by 204-192=12

    // Clear bit 255 (for clamping in X25519)
    h.v[4] &= (1LL << 51) - 1;
}

// Reduce/carry field element to ensure each limb is in range [0, 2^51)
// Also performs full reduction mod p = 2^255 - 19
inline void fe25519_reduce(Fe25519& h) {
    int64_t c;

    // First pass: propagate carries
    c = h.v[0] >> 51; h.v[1] += c; h.v[0] &= FE25519_MASK51;
    c = h.v[1] >> 51; h.v[2] += c; h.v[1] &= FE25519_MASK51;
    c = h.v[2] >> 51; h.v[3] += c; h.v[2] &= FE25519_MASK51;
    c = h.v[3] >> 51; h.v[4] += c; h.v[3] &= FE25519_MASK51;

    // Handle overflow from limb 4: 2^255 ≡ 19 (mod p)
    c = h.v[4] >> 51; h.v[0] += c * 19; h.v[4] &= FE25519_MASK51;

    // Second pass to handle any carries from the multiplication by 19
    c = h.v[0] >> 51; h.v[1] += c; h.v[0] &= FE25519_MASK51;
    c = h.v[1] >> 51; h.v[2] += c; h.v[1] &= FE25519_MASK51;
    c = h.v[2] >> 51; h.v[3] += c; h.v[2] &= FE25519_MASK51;
    c = h.v[3] >> 51; h.v[4] += c; h.v[3] &= FE25519_MASK51;
    c = h.v[4] >> 51; h.v[0] += c * 19; h.v[4] &= FE25519_MASK51;

    // Final full reduction: if h >= p, subtract p
    // p = 2^255 - 19, so we need h - p if h >= p
    // Check if h >= p by checking if h - p >= 0
    // h - p = h - 2^255 + 19 = h[0] + 19 - 2^255 (approximately)

    // Compute h - p (which equals h + 19 - 2^255)
    c = h.v[0] + 19;
    c >>= 51;
    c += h.v[1]; c >>= 51;
    c += h.v[2]; c >>= 51;
    c += h.v[3]; c >>= 51;
    c += h.v[4]; c >>= 51;

    // If c != 0, then h >= p, so subtract p (add 19 and subtract 2^255)
    h.v[0] += 19 * c;

    // Propagate carries again
    c = h.v[0] >> 51; h.v[1] += c; h.v[0] &= FE25519_MASK51;
    c = h.v[1] >> 51; h.v[2] += c; h.v[1] &= FE25519_MASK51;
    c = h.v[2] >> 51; h.v[3] += c; h.v[2] &= FE25519_MASK51;
    c = h.v[3] >> 51; h.v[4] += c; h.v[3] &= FE25519_MASK51;
    h.v[4] &= FE25519_MASK51;
}

// Store field element as 32 bytes (little-endian)
// Performs full reduction first
inline void fe25519_tobytes(uint8_t* s, const Fe25519& h_in) {
    Fe25519 h = h_in;
    fe25519_reduce(h);

    // Pack 5 limbs of 51 bits into 32 bytes
    // Total: 5 * 51 = 255 bits, fits in 32 bytes

    uint64_t t0 = static_cast<uint64_t>(h.v[0]);
    uint64_t t1 = static_cast<uint64_t>(h.v[1]);
    uint64_t t2 = static_cast<uint64_t>(h.v[2]);
    uint64_t t3 = static_cast<uint64_t>(h.v[3]);
    uint64_t t4 = static_cast<uint64_t>(h.v[4]);

    // Bytes 0-7: bits 0-63 (all of limb 0 + 13 bits of limb 1)
    fe25519_store64_le(s, t0 | (t1 << 51));

    // Bytes 8-15: bits 64-127 (38 bits of limb 1 + 26 bits of limb 2)
    fe25519_store64_le(s + 8, (t1 >> 13) | (t2 << 38));

    // Bytes 16-23: bits 128-191 (25 bits of limb 2 + 39 bits of limb 3)
    fe25519_store64_le(s + 16, (t2 >> 26) | (t3 << 25));

    // Bytes 24-31: bits 192-255 (12 bits of limb 3 + 51 bits of limb 4)
    fe25519_store64_le(s + 24, (t3 >> 39) | (t4 << 12));
}

// Addition: h = f + g
inline void fe25519_add(Fe25519& h, const Fe25519& f, const Fe25519& g) {
    h.v[0] = f.v[0] + g.v[0];
    h.v[1] = f.v[1] + g.v[1];
    h.v[2] = f.v[2] + g.v[2];
    h.v[3] = f.v[3] + g.v[3];
    h.v[4] = f.v[4] + g.v[4];
}

// Subtraction: h = f - g
// Add a large multiple of p to ensure result is positive
inline void fe25519_sub(Fe25519& h, const Fe25519& f, const Fe25519& g) {
    // Add 8*p to avoid negative intermediate values
    // Using the same constants as curve25519-donna:
    // 8*p contribution: 2^54 - 152 for limb 0, 2^54 - 8 for other limbs
    // These values are chosen so that after carry propagation, the excess
    // reduces to 0 mod p. The larger constants (2^54 vs 2^52) ensure we have
    // enough headroom for subsequent operations before reduction.
    constexpr int64_t EIGHT_P0 = (1LL << 54) - 152;  // 8*p contribution to limb 0
    constexpr int64_t EIGHT_P = (1LL << 54) - 8;     // 8*p contribution to other limbs

    h.v[0] = f.v[0] + EIGHT_P0 - g.v[0];
    h.v[1] = f.v[1] + EIGHT_P - g.v[1];
    h.v[2] = f.v[2] + EIGHT_P - g.v[2];
    h.v[3] = f.v[3] + EIGHT_P - g.v[3];
    h.v[4] = f.v[4] + EIGHT_P - g.v[4];
}

// Internal carry function for multiplication results
inline void fe25519_carry(Fe25519& h) {
    int64_t c;
    c = h.v[0] >> 51; h.v[1] += c; h.v[0] &= FE25519_MASK51;
    c = h.v[1] >> 51; h.v[2] += c; h.v[1] &= FE25519_MASK51;
    c = h.v[2] >> 51; h.v[3] += c; h.v[2] &= FE25519_MASK51;
    c = h.v[3] >> 51; h.v[4] += c; h.v[3] &= FE25519_MASK51;
    c = h.v[4] >> 51; h.v[0] += c * 19; h.v[4] &= FE25519_MASK51;

    // One more round to handle carry from the *19
    c = h.v[0] >> 51; h.v[1] += c; h.v[0] &= FE25519_MASK51;
}

// Multiplication: h = f * g
// Uses the identity 2^255 ≡ 19 (mod p) for reduction
inline void fe25519_mul(Fe25519& h, const Fe25519& f, const Fe25519& g) {
    // Schoolbook multiplication with reduction
    // f = f0 + f1*2^51 + f2*2^102 + f3*2^153 + f4*2^204
    // g = g0 + g1*2^51 + g2*2^102 + g3*2^153 + g4*2^204
    //
    // Products that exceed 2^255 are reduced using 2^255 ≡ 19 (mod p)
    // 2^(255+k) ≡ 19 * 2^k (mod p)
    //
    // For radix 2^51: 2^(51*5) = 2^255 ≡ 19
    // So f4*g1*2^(51*5) = f4*g1*2^255 ≡ 19*f4*g1 (contributes to coefficient of 2^51)

    int64_t f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    int64_t g0 = g.v[0], g1 = g.v[1], g2 = g.v[2], g3 = g.v[3], g4 = g.v[4];

    // Precompute 19 * gi for reduction
    int64_t g1_19 = 19 * g1;
    int64_t g2_19 = 19 * g2;
    int64_t g3_19 = 19 * g3;
    int64_t g4_19 = 19 * g4;

    // Compute product coefficients using __int128 for full precision
    // h0 = f0*g0 + 19*(f1*g4 + f2*g3 + f3*g2 + f4*g1)
    // h1 = f0*g1 + f1*g0 + 19*(f2*g4 + f3*g3 + f4*g2)
    // h2 = f0*g2 + f1*g1 + f2*g0 + 19*(f3*g4 + f4*g3)
    // h3 = f0*g3 + f1*g2 + f2*g1 + f3*g0 + 19*(f4*g4)
    // h4 = f0*g4 + f1*g3 + f2*g2 + f3*g1 + f4*g0

    __int128 h0 = static_cast<__int128>(f0) * g0
                + static_cast<__int128>(f1) * g4_19
                + static_cast<__int128>(f2) * g3_19
                + static_cast<__int128>(f3) * g2_19
                + static_cast<__int128>(f4) * g1_19;

    __int128 h1 = static_cast<__int128>(f0) * g1
                + static_cast<__int128>(f1) * g0
                + static_cast<__int128>(f2) * g4_19
                + static_cast<__int128>(f3) * g3_19
                + static_cast<__int128>(f4) * g2_19;

    __int128 h2 = static_cast<__int128>(f0) * g2
                + static_cast<__int128>(f1) * g1
                + static_cast<__int128>(f2) * g0
                + static_cast<__int128>(f3) * g4_19
                + static_cast<__int128>(f4) * g3_19;

    __int128 h3 = static_cast<__int128>(f0) * g3
                + static_cast<__int128>(f1) * g2
                + static_cast<__int128>(f2) * g1
                + static_cast<__int128>(f3) * g0
                + static_cast<__int128>(f4) * g4_19;

    __int128 h4 = static_cast<__int128>(f0) * g4
                + static_cast<__int128>(f1) * g3
                + static_cast<__int128>(f2) * g2
                + static_cast<__int128>(f3) * g1
                + static_cast<__int128>(f4) * g0;

    // Carry and reduce
    // Each hi can be up to 5 * 2^51 * 2^51 * 19 ≈ 2^110, fits in __int128

    __int128 c;
    c = h0 >> 51; h1 += c; h0 &= FE25519_MASK51;
    c = h1 >> 51; h2 += c; h1 &= FE25519_MASK51;
    c = h2 >> 51; h3 += c; h2 &= FE25519_MASK51;
    c = h3 >> 51; h4 += c; h3 &= FE25519_MASK51;
    c = h4 >> 51; h0 += c * 19; h4 &= FE25519_MASK51;

    // One more carry round
    c = h0 >> 51; h1 += c; h0 &= FE25519_MASK51;

    h.v[0] = static_cast<int64_t>(h0);
    h.v[1] = static_cast<int64_t>(h1);
    h.v[2] = static_cast<int64_t>(h2);
    h.v[3] = static_cast<int64_t>(h3);
    h.v[4] = static_cast<int64_t>(h4);
}

// Squaring: h = f^2 (optimized version of multiplication)
inline void fe25519_sq(Fe25519& h, const Fe25519& f) {
    // Squaring can be optimized because f_i * f_j = f_j * f_i
    // So we can compute these products once and double them

    int64_t f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];

    // Doubled products for cross terms
    int64_t f0_2 = 2 * f0;
    int64_t f1_2 = 2 * f1;
    int64_t f2_2 = 2 * f2;
    int64_t f3_2 = 2 * f3;

    // Precompute 19 * fi for reduction
    int64_t f1_38 = 38 * f1;  // 2 * 19 * f1 for doubled cross terms
    int64_t f2_38 = 38 * f2;
    int64_t f3_38 = 38 * f3;
    int64_t f4_19 = 19 * f4;
    int64_t f4_38 = 38 * f4;

    // h0 = f0^2 + 2*19*(f1*f4 + f2*f3)
    // h1 = 2*f0*f1 + 2*19*(f2*f4) + 19*f3^2
    // h2 = 2*f0*f2 + f1^2 + 2*19*(f3*f4)
    // h3 = 2*f0*f3 + 2*f1*f2 + 19*f4^2
    // h4 = 2*f0*f4 + 2*f1*f3 + f2^2

    __int128 h0 = static_cast<__int128>(f0) * f0
                + static_cast<__int128>(f1_38) * f4
                + static_cast<__int128>(f2_38) * f3;

    __int128 h1 = static_cast<__int128>(f0_2) * f1
                + static_cast<__int128>(f2_38) * f4
                + static_cast<__int128>(f3) * f3_38 / 2;  // 19 * f3^2

    // Recompute h1 properly: 19 * f3^2
    h1 = static_cast<__int128>(f0_2) * f1
       + static_cast<__int128>(f2_38) * f4
       + static_cast<__int128>(19) * f3 * f3;

    __int128 h2 = static_cast<__int128>(f0_2) * f2
                + static_cast<__int128>(f1) * f1
                + static_cast<__int128>(f3_38) * f4;

    __int128 h3 = static_cast<__int128>(f0_2) * f3
                + static_cast<__int128>(f1_2) * f2
                + static_cast<__int128>(f4_19) * f4;

    __int128 h4 = static_cast<__int128>(f0_2) * f4
                + static_cast<__int128>(f1_2) * f3
                + static_cast<__int128>(f2) * f2;

    // Carry and reduce
    __int128 c;
    c = h0 >> 51; h1 += c; h0 &= FE25519_MASK51;
    c = h1 >> 51; h2 += c; h1 &= FE25519_MASK51;
    c = h2 >> 51; h3 += c; h2 &= FE25519_MASK51;
    c = h3 >> 51; h4 += c; h3 &= FE25519_MASK51;
    c = h4 >> 51; h0 += c * 19; h4 &= FE25519_MASK51;

    // One more carry round
    c = h0 >> 51; h1 += c; h0 &= FE25519_MASK51;

    h.v[0] = static_cast<int64_t>(h0);
    h.v[1] = static_cast<int64_t>(h1);
    h.v[2] = static_cast<int64_t>(h2);
    h.v[3] = static_cast<int64_t>(h3);
    h.v[4] = static_cast<int64_t>(h4);
}

// Square n times: h = f^(2^n)
inline void fe25519_sq_n(Fe25519& h, const Fe25519& f, int n) {
    fe25519_sq(h, f);
    for (int i = 1; i < n; i++) {
        fe25519_sq(h, h);
    }
}

// Inversion: out = z^(-1) = z^(p-2) using Fermat's little theorem
// p - 2 = 2^255 - 21 = 2^255 - 16 - 4 - 1 = (2^255 - 19) - 2
// We use an addition chain for efficient exponentiation
inline void fe25519_invert(Fe25519& out, const Fe25519& z) {
    Fe25519 t0, t1, t2, t3;

    // Using the addition chain for p-2 = 2^255 - 21
    // This is a standard addition chain used in ref10 implementation

    // t0 = z^2
    fe25519_sq(t0, z);

    // t1 = z^4
    fe25519_sq(t1, t0);
    // t1 = z^8
    fe25519_sq(t1, t1);

    // t1 = z^9 = z^8 * z
    fe25519_mul(t1, t1, z);

    // t0 = z^11 = z^9 * z^2
    fe25519_mul(t0, t1, t0);

    // t2 = z^22 = (z^11)^2
    fe25519_sq(t2, t0);

    // t1 = z^31 = z^22 * z^9
    fe25519_mul(t1, t2, t1);

    // t2 = z^(2^5 * 31) = z^(31 * 32) = z^992
    fe25519_sq_n(t2, t1, 5);

    // t1 = z^1023 = z^992 * z^31
    fe25519_mul(t1, t2, t1);

    // t2 = z^(2^10 * 1023) = z^(1023 * 1024)
    fe25519_sq_n(t2, t1, 10);

    // t2 = z^(2^10 * 1023 + 1023) = z^(1047552 + 1023)
    fe25519_mul(t2, t2, t1);

    // t3 = z^(2^20 * above)
    fe25519_sq_n(t3, t2, 20);

    // t2 = t3 * t2
    fe25519_mul(t2, t3, t2);

    // t2 = t2^(2^10)
    fe25519_sq_n(t2, t2, 10);

    // t1 = t2 * t1
    fe25519_mul(t1, t2, t1);

    // t2 = t1^(2^50)
    fe25519_sq_n(t2, t1, 50);

    // t2 = t2 * t1
    fe25519_mul(t2, t2, t1);

    // t3 = t2^(2^100)
    fe25519_sq_n(t3, t2, 100);

    // t2 = t3 * t2
    fe25519_mul(t2, t3, t2);

    // t2 = t2^(2^50)
    fe25519_sq_n(t2, t2, 50);

    // t1 = t2 * t1
    fe25519_mul(t1, t2, t1);

    // t1 = t1^(2^5) = t1^32
    fe25519_sq_n(t1, t1, 5);

    // out = t1 * t0 = z^(2^255 - 21)
    fe25519_mul(out, t1, t0);
}

// Compute z^((p-5)/8) = z^(2^252 - 3)
// Used for computing square roots on Curve25519
// (p-5)/8 = (2^255 - 19 - 5)/8 = (2^255 - 24)/8 = 2^252 - 3
inline void fe25519_pow2523(Fe25519& out, const Fe25519& z) {
    Fe25519 t0, t1, t2;

    // Addition chain for 2^252 - 3

    // t0 = z^2
    fe25519_sq(t0, z);

    // t1 = z^4
    fe25519_sq(t1, t0);
    // t1 = z^8
    fe25519_sq(t1, t1);

    // t1 = z^9 = z^8 * z
    fe25519_mul(t1, t1, z);

    // t0 = z^11 = z^9 * z^2
    fe25519_mul(t0, t1, t0);

    // t0 = z^22 = (z^11)^2
    fe25519_sq(t0, t0);

    // t0 = z^31 = z^22 * z^9
    fe25519_mul(t0, t0, t1);

    // t1 = z^(31 * 2^5) = z^992
    fe25519_sq_n(t1, t0, 5);

    // t0 = z^1023 = z^992 * z^31
    fe25519_mul(t0, t1, t0);

    // t1 = z^(1023 * 2^10)
    fe25519_sq_n(t1, t0, 10);

    // t1 = z^(1023 * 2^10 + 1023)
    fe25519_mul(t1, t1, t0);

    // t2 = z^(above * 2^20)
    fe25519_sq_n(t2, t1, 20);

    // t1 = t2 * t1
    fe25519_mul(t1, t2, t1);

    // t1 = t1^(2^10)
    fe25519_sq_n(t1, t1, 10);

    // t0 = t1 * t0
    fe25519_mul(t0, t1, t0);

    // t1 = t0^(2^50)
    fe25519_sq_n(t1, t0, 50);

    // t1 = t1 * t0
    fe25519_mul(t1, t1, t0);

    // t2 = t1^(2^100)
    fe25519_sq_n(t2, t1, 100);

    // t1 = t2 * t1
    fe25519_mul(t1, t2, t1);

    // t1 = t1^(2^50)
    fe25519_sq_n(t1, t1, 50);

    // t0 = t1 * t0
    fe25519_mul(t0, t1, t0);

    // t0 = t0^(2^2) = t0^4
    fe25519_sq_n(t0, t0, 2);

    // out = t0 * z = z^(2^252 - 3)
    fe25519_mul(out, t0, z);
}

// Set h to 0
inline void fe25519_zero(Fe25519& h) {
    h.v[0] = h.v[1] = h.v[2] = h.v[3] = h.v[4] = 0;
}

// Set h to 1
inline void fe25519_one(Fe25519& h) {
    h.v[0] = 1;
    h.v[1] = h.v[2] = h.v[3] = h.v[4] = 0;
}

// Copy: h = f
inline void fe25519_copy(Fe25519& h, const Fe25519& f) {
    h.v[0] = f.v[0];
    h.v[1] = f.v[1];
    h.v[2] = f.v[2];
    h.v[3] = f.v[3];
    h.v[4] = f.v[4];
}

// Conditional swap: swap f and g if b == 1
// Constant-time implementation
inline void fe25519_cswap(Fe25519& f, Fe25519& g, int64_t b) {
    b = -b;  // 0 or -1 (all bits set)
    for (int i = 0; i < 5; i++) {
        int64_t x = (f.v[i] ^ g.v[i]) & b;
        f.v[i] ^= x;
        g.v[i] ^= x;
    }
}

// Conditional move: h = g if b == 1, else h unchanged
// Constant-time implementation
inline void fe25519_cmov(Fe25519& h, const Fe25519& g, int64_t b) {
    b = -b;  // 0 or -1 (all bits set)
    for (int i = 0; i < 5; i++) {
        h.v[i] ^= (h.v[i] ^ g.v[i]) & b;
    }
}

// Negate: h = -f
inline void fe25519_neg(Fe25519& h, const Fe25519& f) {
    Fe25519 zero;
    fe25519_zero(zero);
    fe25519_sub(h, zero, f);
}

// Check if f is negative (least significant bit after reduction)
inline int fe25519_isnegative(const Fe25519& f) {
    uint8_t s[32];
    fe25519_tobytes(s, f);
    return s[0] & 1;
}

// Check if f is zero
inline int fe25519_iszero(const Fe25519& f) {
    uint8_t s[32];
    fe25519_tobytes(s, f);
    uint8_t r = 0;
    for (int i = 0; i < 32; i++) {
        r |= s[i];
    }
    return (r == 0) ? 1 : 0;
}

// Absolute value: h = |f| (make f positive, i.e., clear sign bit)
inline void fe25519_abs(Fe25519& h, const Fe25519& f) {
    fe25519_copy(h, f);
    Fe25519 neg_f;
    fe25519_neg(neg_f, f);
    fe25519_cmov(h, neg_f, fe25519_isnegative(f));
}

// ============================================================================
// X25519 (RFC 7748) Scalar Multiplication
// ============================================================================

// Constant a24 = (A - 2) / 4 = (486662 - 2) / 4 = 121665 for Curve25519
// where A is the Montgomery coefficient in y^2 = x^3 + Ax^2 + x
// Note: Some sources incorrectly state (A+2)/4, but RFC 7748 specifies (A-2)/4
constexpr int64_t X25519_A24 = 121665;

// Scalar clamping for X25519 (RFC 7748, Section 5)
// Clamps the scalar to ensure it is a valid X25519 private key:
// - Clear bits 0, 1, 2 (make it a multiple of 8, for cofactor)
// - Clear bit 255 (ensure scalar < 2^255)
// - Set bit 254 (ensure constant-time ladder execution)
inline void x25519_clamp(uint8_t* k) {
    k[0] &= 248;   // Clear bits 0, 1, 2 (248 = 0xF8)
    k[31] &= 127;  // Clear bit 255 (127 = 0x7F)
    k[31] |= 64;   // Set bit 254 (64 = 0x40)
}

// Multiply field element by the constant a24 = 121666
// h = a24 * f
inline void fe25519_mul_a24(Fe25519& h, const Fe25519& f) {
    __int128 h0 = static_cast<__int128>(X25519_A24) * f.v[0];
    __int128 h1 = static_cast<__int128>(X25519_A24) * f.v[1];
    __int128 h2 = static_cast<__int128>(X25519_A24) * f.v[2];
    __int128 h3 = static_cast<__int128>(X25519_A24) * f.v[3];
    __int128 h4 = static_cast<__int128>(X25519_A24) * f.v[4];

    // Carry and reduce
    __int128 c;
    c = h0 >> 51; h1 += c; h0 &= FE25519_MASK51;
    c = h1 >> 51; h2 += c; h1 &= FE25519_MASK51;
    c = h2 >> 51; h3 += c; h2 &= FE25519_MASK51;
    c = h3 >> 51; h4 += c; h3 &= FE25519_MASK51;
    c = h4 >> 51; h0 += c * 19; h4 &= FE25519_MASK51;

    // One more carry round
    c = h0 >> 51; h1 += c; h0 &= FE25519_MASK51;

    h.v[0] = static_cast<int64_t>(h0);
    h.v[1] = static_cast<int64_t>(h1);
    h.v[2] = static_cast<int64_t>(h2);
    h.v[3] = static_cast<int64_t>(h3);
    h.v[4] = static_cast<int64_t>(h4);
}

// Montgomery ladder step - differential addition
// Given points P = (x2 : z2) and Q = (x3 : z3) = P + base, computes:
//   P' = 2P and Q' = P + Q (differential addition)
// using only the x-coordinate of the base point
//
// Montgomery ladder formulas (x-coordinate only):
//   A = x2 + z2
//   AA = A^2
//   B = x2 - z2
//   BB = B^2
//   E = AA - BB
//   C = x3 + z3
//   D = x3 - z3
//   DA = D * A
//   CB = C * B
//   x3 = (DA + CB)^2
//   z3 = x1 * (DA - CB)^2
//   x2 = AA * BB
//   z2 = E * (AA + a24 * E)
//
// where a24 = (A + 2) / 4 = 121666 for Curve25519
inline void x25519_ladder_step(
    Fe25519& x2, Fe25519& z2,  // Point P (updated to 2P)
    Fe25519& x3, Fe25519& z3,  // Point Q = P + base (updated to P + Q)
    const Fe25519& x1          // x-coordinate of base point
) {
    Fe25519 A, AA, B, BB, E, C, D, DA, CB, t0, t1;

    // A = x2 + z2
    fe25519_add(A, x2, z2);
    // AA = A^2
    fe25519_sq(AA, A);
    // B = x2 - z2
    fe25519_sub(B, x2, z2);
    // BB = B^2
    fe25519_sq(BB, B);
    // E = AA - BB
    fe25519_sub(E, AA, BB);
    // C = x3 + z3
    fe25519_add(C, x3, z3);
    // D = x3 - z3
    fe25519_sub(D, x3, z3);
    // DA = D * A
    fe25519_mul(DA, D, A);
    // CB = C * B
    fe25519_mul(CB, C, B);

    // x3 = (DA + CB)^2
    fe25519_add(t0, DA, CB);
    fe25519_sq(x3, t0);

    // z3 = x1 * (DA - CB)^2
    fe25519_sub(t0, DA, CB);
    fe25519_sq(t1, t0);
    fe25519_mul(z3, x1, t1);

    // x2 = AA * BB
    fe25519_mul(x2, AA, BB);

    // z2 = E * (AA + a24 * E)
    fe25519_mul_a24(t0, E);      // t0 = a24 * E
    fe25519_add(t0, AA, t0);     // t0 = AA + a24 * E
    fe25519_mul(z2, E, t0);      // z2 = E * (AA + a24 * E)
}

// Full X25519 scalar multiplication: q = n * p
// Computes the x-coordinate of n*P where P has x-coordinate p
// Uses the Montgomery ladder for constant-time execution
//
// Input:
//   n: 32-byte scalar (will be clamped internally)
//   p: 32-byte x-coordinate of input point
// Output:
//   q: 32-byte x-coordinate of result point
inline void x25519_scalarmult(
    uint8_t* q,        // 32-byte output
    const uint8_t* n,  // 32-byte scalar
    const uint8_t* p   // 32-byte point (x-coordinate)
) {
    // Copy and clamp the scalar
    uint8_t e[32];
    for (int i = 0; i < 32; i++) {
        e[i] = n[i];
    }
    x25519_clamp(e);

    // Load the base point x-coordinate
    Fe25519 x1;
    fe25519_frombytes(x1, p);

    // Initialize ladder state:
    // (x2 : z2) = (1 : 0) = point at infinity
    // (x3 : z3) = (x1 : 1) = base point
    Fe25519 x2, z2, x3, z3;
    fe25519_one(x2);
    fe25519_zero(z2);
    fe25519_copy(x3, x1);
    fe25519_one(z3);

    // Montgomery ladder: process bits from 254 down to 0
    // (bit 255 is cleared by clamping, bit 254 is set)
    int64_t swap = 0;
    for (int i = 254; i >= 0; i--) {
        // Extract bit i of the scalar
        int64_t bit = (e[i >> 3] >> (i & 7)) & 1;

        // Conditional swap based on bit XOR previous swap state
        swap ^= bit;
        fe25519_cswap(x2, x3, swap);
        fe25519_cswap(z2, z3, swap);
        swap = bit;

        // Ladder step: (x2, z2) = 2*(x2, z2), (x3, z3) = (x2, z2) + (x3, z3)
        x25519_ladder_step(x2, z2, x3, z3, x1);
    }

    // Final conditional swap
    fe25519_cswap(x2, x3, swap);
    fe25519_cswap(z2, z3, swap);

    // Compute result: x = x2 * z2^(-1)
    Fe25519 z2_inv, result;
    fe25519_invert(z2_inv, z2);
    fe25519_mul(result, x2, z2_inv);

    // Store result
    fe25519_tobytes(q, result);
}

// X25519 base point multiplication: q = n * G
// Computes the x-coordinate of n*G where G is the standard base point
// with x-coordinate 9
//
// Input:
//   n: 32-byte scalar
// Output:
//   q: 32-byte x-coordinate of result point
inline void x25519_scalarmult_base(
    uint8_t* q,        // 32-byte output
    const uint8_t* n   // 32-byte scalar
) {
    // Standard base point for Curve25519 has x-coordinate = 9
    static const uint8_t basepoint[32] = {
        9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    x25519_scalarmult(q, n, basepoint);
}

}  // namespace torchscience::kernel::encryption
