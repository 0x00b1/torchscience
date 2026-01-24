// src/torchscience/csrc/kernel/encryption/galois_field.h
//
// GF(2^8) Galois Field arithmetic for cryptographic applications.
// Uses the AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
//
// This implementation provides efficient multiplication via log/exp tables,
// which is essential for Shamir secret sharing and Reed-Solomon codes.

#pragma once

#include <cstdint>
#include <array>

namespace torchscience::kernel::encryption {

// AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1
// The low 8 bits (0x1B) are used for reduction when the high bit is set
constexpr uint16_t GF256_POLYNOMIAL = 0x11B;
constexpr uint8_t GF256_REDUCTION = 0x1B;

// Generator element for GF(2^8) - primitive element that generates all non-zero elements
constexpr uint8_t GF256_GENERATOR = 0x03;

namespace detail {

// Compile-time computation of GF(2^8) multiplication without tables
// Uses the "Russian peasant" algorithm (shift and conditional XOR)
constexpr uint8_t gf256_mul_slow(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    uint8_t temp_a = a;
    uint8_t temp_b = b;

    for (int i = 0; i < 8; i++) {
        if (temp_b & 1) {
            result ^= temp_a;
        }
        bool high_bit = (temp_a & 0x80) != 0;
        temp_a <<= 1;
        if (high_bit) {
            temp_a ^= GF256_REDUCTION;
        }
        temp_b >>= 1;
    }
    return result;
}

// Compile-time generation of exp table
// exp_table[i] = g^i where g = 0x03 (generator)
// exp_table[255] = exp_table[0] = 1 for wraparound convenience
constexpr std::array<uint8_t, 512> generate_exp_table() {
    std::array<uint8_t, 512> table{};
    uint8_t val = 1;
    for (int i = 0; i < 256; i++) {
        table[i] = val;
        table[i + 256] = val;  // Duplicate for easy modular arithmetic
        val = gf256_mul_slow(val, GF256_GENERATOR);
    }
    return table;
}

// Compile-time generation of log table
// log_table[g^i] = i for i in [0, 254]
// log_table[0] is undefined (log of 0 doesn't exist), set to 0 as sentinel
constexpr std::array<uint8_t, 256> generate_log_table() {
    std::array<uint8_t, 256> table{};
    uint8_t val = 1;
    table[0] = 0;  // Sentinel value, log(0) is undefined
    for (int i = 0; i < 255; i++) {
        table[val] = static_cast<uint8_t>(i);
        val = gf256_mul_slow(val, GF256_GENERATOR);
    }
    return table;
}

}  // namespace detail

// Pre-computed lookup tables for efficient GF(2^8) arithmetic
// exp_table[i] = 3^i mod P(x), duplicated for easy modular reduction
constexpr std::array<uint8_t, 512> GF256_EXP_TABLE = detail::generate_exp_table();

// log_table[x] = i such that 3^i = x, undefined for x = 0
constexpr std::array<uint8_t, 256> GF256_LOG_TABLE = detail::generate_log_table();

// Addition in GF(2^8) is simply XOR
// a + b = a XOR b
inline constexpr uint8_t gf_add(uint8_t a, uint8_t b) {
    return a ^ b;
}

// Subtraction in GF(2^8) is the same as addition (XOR)
// a - b = a XOR b (since -1 = 1 in characteristic 2)
inline constexpr uint8_t gf_sub(uint8_t a, uint8_t b) {
    return a ^ b;
}

// Multiplication in GF(2^8) using log/exp tables
// a * b = exp(log(a) + log(b))
// Special case: if a = 0 or b = 0, result is 0
inline uint8_t gf_multiply(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    // Use the duplicate exp table to avoid modular arithmetic
    // log(a) + log(b) can be at most 254 + 254 = 508 < 512
    int log_sum = GF256_LOG_TABLE[a] + GF256_LOG_TABLE[b];
    return GF256_EXP_TABLE[log_sum];
}

// Multiplicative inverse in GF(2^8) using log/exp tables
// a^(-1) = exp(255 - log(a)) since a^255 = 1 for all non-zero a
// Special case: inverse of 0 is undefined, returns 0 as error indicator
inline uint8_t gf_inverse(uint8_t a) {
    if (a == 0) {
        return 0;  // Undefined, but return 0 as sentinel
    }
    // a^(-1) = a^254 = exp(255 - log(a))
    int log_inv = 255 - GF256_LOG_TABLE[a];
    return GF256_EXP_TABLE[log_inv];
}

// Division in GF(2^8)
// a / b = a * b^(-1) = exp(log(a) - log(b))
// Special case: division by 0 returns 0 (undefined)
inline uint8_t gf_divide(uint8_t a, uint8_t b) {
    if (b == 0) {
        return 0;  // Division by zero, undefined
    }
    if (a == 0) {
        return 0;
    }
    // Use the duplicate exp table: if log(a) < log(b), add 255
    int log_diff = GF256_LOG_TABLE[a] - GF256_LOG_TABLE[b];
    if (log_diff < 0) {
        log_diff += 255;
    }
    return GF256_EXP_TABLE[log_diff];
}

// Exponentiation in GF(2^8) using log/exp tables
// a^n = exp(n * log(a) mod 255)
// Special case: 0^n = 0 for n > 0, 0^0 = 1, a^0 = 1
inline uint8_t gf_power(uint8_t a, uint8_t n) {
    if (n == 0) {
        return 1;  // Anything to the power 0 is 1
    }
    if (a == 0) {
        return 0;  // 0 to any positive power is 0
    }
    int log_result = (static_cast<int>(GF256_LOG_TABLE[a]) * n) % 255;
    return GF256_EXP_TABLE[log_result];
}

// Evaluate polynomial at point x in GF(2^8) using Horner's method
// coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ... + coeffs[degree]*x^degree
// Coefficients are stored in ascending order of degree
inline uint8_t gf_poly_eval(const uint8_t* coeffs, int degree, uint8_t x) {
    if (degree < 0) {
        return 0;
    }

    // Horner's method: evaluate from highest to lowest degree
    // p(x) = (...((a_n)*x + a_{n-1})*x + ... + a_1)*x + a_0
    uint8_t result = coeffs[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = gf_add(gf_multiply(result, x), coeffs[i]);
    }
    return result;
}

// Evaluate polynomial at point x in GF(2^8) - version with array
// This is a convenience wrapper for fixed-size arrays
template <size_t N>
inline uint8_t gf_poly_eval(const std::array<uint8_t, N>& coeffs, uint8_t x) {
    return gf_poly_eval(coeffs.data(), static_cast<int>(N) - 1, x);
}

// Lagrange interpolation at x=0 in GF(2^8)
// Given k points (x_i, y_i), reconstruct the polynomial and evaluate at x=0
// This is the core operation for Shamir secret sharing reconstruction
//
// shares: array of y values (share values)
// indices: array of x values (share indices, must be non-zero and distinct)
// k: number of shares (threshold)
//
// Returns: the interpolated value at x=0 (the secret)
//
// Formula: f(0) = sum_{i=0}^{k-1} y_i * prod_{j!=i} (x_j / (x_j - x_i))
//
// Note: For Shamir secret sharing, x values are typically 1, 2, 3, ...
// and cannot be 0 (since 0 is where the secret is stored)
inline uint8_t gf_lagrange_interpolate(
    const uint8_t* shares,
    const uint8_t* indices,
    int k
) {
    uint8_t result = 0;

    for (int i = 0; i < k; i++) {
        // Calculate Lagrange basis polynomial L_i(0)
        // L_i(0) = prod_{j!=i} (0 - x_j) / (x_i - x_j)
        //        = prod_{j!=i} x_j / (x_i - x_j)
        //        = prod_{j!=i} x_j / (x_i + x_j)  [since subtraction = addition in GF(2)]

        uint8_t numerator = 1;
        uint8_t denominator = 1;

        for (int j = 0; j < k; j++) {
            if (i != j) {
                // In GF(2^8): 0 - x_j = x_j (additive inverse is same as element)
                numerator = gf_multiply(numerator, indices[j]);
                // x_i - x_j = x_i + x_j = x_i XOR x_j
                denominator = gf_multiply(denominator, gf_add(indices[i], indices[j]));
            }
        }

        // L_i(0) = numerator / denominator
        uint8_t basis = gf_divide(numerator, denominator);

        // Add y_i * L_i(0) to result
        result = gf_add(result, gf_multiply(shares[i], basis));
    }

    return result;
}

// Lagrange interpolation at arbitrary point in GF(2^8)
// Evaluates the interpolated polynomial at point x
inline uint8_t gf_lagrange_interpolate_at(
    const uint8_t* shares,
    const uint8_t* indices,
    int k,
    uint8_t x
) {
    uint8_t result = 0;

    for (int i = 0; i < k; i++) {
        // Calculate Lagrange basis polynomial L_i(x)
        // L_i(x) = prod_{j!=i} (x - x_j) / (x_i - x_j)

        uint8_t numerator = 1;
        uint8_t denominator = 1;

        for (int j = 0; j < k; j++) {
            if (i != j) {
                numerator = gf_multiply(numerator, gf_sub(x, indices[j]));
                denominator = gf_multiply(denominator, gf_sub(indices[i], indices[j]));
            }
        }

        uint8_t basis = gf_divide(numerator, denominator);
        result = gf_add(result, gf_multiply(shares[i], basis));
    }

    return result;
}

// Compute Lagrange basis coefficients for a set of x values
// This precomputes the denominators for efficiency when doing multiple interpolations
// with the same x values but different y values
//
// indices: array of x values
// k: number of points
// basis_coeffs: output array of k basis denominator inverses
//
// After calling, basis_coeffs[i] = 1 / prod_{j!=i} (x_i - x_j)
inline void gf_lagrange_precompute(
    const uint8_t* indices,
    int k,
    uint8_t* basis_coeffs
) {
    for (int i = 0; i < k; i++) {
        uint8_t denominator = 1;
        for (int j = 0; j < k; j++) {
            if (i != j) {
                denominator = gf_multiply(denominator, gf_sub(indices[i], indices[j]));
            }
        }
        basis_coeffs[i] = gf_inverse(denominator);
    }
}

// Fast Lagrange interpolation at x=0 with precomputed basis coefficients
// shares: array of y values
// indices: array of x values
// basis_coeffs: precomputed basis denominator inverses from gf_lagrange_precompute
// k: number of points
inline uint8_t gf_lagrange_interpolate_fast(
    const uint8_t* shares,
    const uint8_t* indices,
    const uint8_t* basis_coeffs,
    int k
) {
    uint8_t result = 0;

    for (int i = 0; i < k; i++) {
        // L_i(0) = basis_coeffs[i] * prod_{j!=i} x_j
        uint8_t numerator = 1;
        for (int j = 0; j < k; j++) {
            if (i != j) {
                numerator = gf_multiply(numerator, indices[j]);
            }
        }

        uint8_t basis = gf_multiply(numerator, basis_coeffs[i]);
        result = gf_add(result, gf_multiply(shares[i], basis));
    }

    return result;
}

// Generate polynomial coefficients for Lagrange basis L_i(x)
// Useful for debugging or when you need the actual polynomial
// coeffs: output array of size k for coefficients
// indices: array of x values
// i: index of basis polynomial to generate
// k: number of points
inline void gf_lagrange_basis_poly(
    uint8_t* coeffs,
    const uint8_t* indices,
    int i,
    int k
) {
    // Initialize: polynomial starts as 1
    for (int j = 0; j < k; j++) {
        coeffs[j] = 0;
    }
    coeffs[0] = 1;

    // Build polynomial by multiplying (x - x_j) for j != i
    // Each multiplication increases degree by 1
    int degree = 0;
    uint8_t denominator = 1;

    for (int j = 0; j < k; j++) {
        if (i != j) {
            // Multiply current polynomial by (x - x_j) = (x + x_j) in GF(2)
            // New coefficients: new[k] = old[k-1] + x_j * old[k]
            degree++;

            // Process from highest to lowest to avoid overwriting
            for (int d = degree; d >= 1; d--) {
                coeffs[d] = gf_add(coeffs[d - 1], gf_multiply(indices[j], coeffs[d]));
            }
            coeffs[0] = gf_multiply(indices[j], coeffs[0]);

            // Accumulate denominator
            denominator = gf_multiply(denominator, gf_sub(indices[i], indices[j]));
        }
    }

    // Divide all coefficients by denominator
    uint8_t inv_denom = gf_inverse(denominator);
    for (int j = 0; j <= degree; j++) {
        coeffs[j] = gf_multiply(coeffs[j], inv_denom);
    }
}

}  // namespace torchscience::kernel::encryption
