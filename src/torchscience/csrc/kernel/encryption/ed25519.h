// src/torchscience/csrc/kernel/encryption/ed25519.h
#pragma once

#include "curve25519.h"

namespace torchscience::kernel::encryption {

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
        1859910466990406LL,
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

}  // namespace torchscience::kernel::encryption
