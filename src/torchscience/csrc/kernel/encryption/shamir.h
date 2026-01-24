// src/torchscience/csrc/kernel/encryption/shamir.h
//
// Shamir's Secret Sharing scheme implementation over GF(2^8).
//
// This implements (k, n) threshold secret sharing where:
// - A secret is split into n shares
// - Any k shares can reconstruct the secret
// - Fewer than k shares reveal no information about the secret
//
// The scheme uses polynomial interpolation in GF(2^8):
// - Split: Create a random polynomial of degree k-1 with secret as constant term
//          Evaluate at x = 1, 2, ..., n to generate n shares
// - Reconstruct: Use Lagrange interpolation to recover the polynomial at x = 0

#pragma once

#include "galois_field.h"
#include <cstdint>

namespace torchscience::kernel::encryption {

// Split secret into n shares with threshold k
//
// For each byte of the secret, constructs a polynomial:
//   f(x) = secret[i] + r[0]*x + r[1]*x^2 + ... + r[k-2]*x^(k-1)
// where r[j] are random coefficients from the randomness array.
//
// Then evaluates f(x) at x = 1, 2, ..., n to generate n shares.
//
// Parameters:
//   secret:     Input secret bytes, length = secret_len
//   shares:     Output array of n shares, each of length secret_len
//               Layout: shares[share_idx * secret_len + byte_idx]
//   randomness: Random coefficients, size = (k-1) * secret_len
//               Layout: randomness[coeff_idx * secret_len + byte_idx]
//               where coeff_idx is 0 to k-2 (for x^1 to x^(k-1))
//   secret_len: Length of the secret in bytes
//   n:          Number of shares to generate (must be <= 255)
//   k:          Threshold - minimum shares needed to reconstruct (2 <= k <= n)
//
// Note: Share indices are 1-indexed (x = 1, 2, ..., n) since x = 0 is
// where the secret is stored in the polynomial.
template <typename scalar_t>
void shamir_split_kernel(
    const scalar_t* secret,
    scalar_t* shares,
    const scalar_t* randomness,
    int64_t secret_len,
    int64_t n,
    int64_t k
) {
    // For each byte position in the secret
    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        // Build the polynomial coefficients for this byte
        // coeffs[0] = secret byte (constant term)
        // coeffs[1..k-1] = random coefficients
        uint8_t secret_byte = static_cast<uint8_t>(secret[byte_idx]);

        // Evaluate polynomial at x = 1, 2, ..., n to generate each share
        for (int64_t share_idx = 0; share_idx < n; share_idx++) {
            // x value for this share (1-indexed)
            uint8_t x = static_cast<uint8_t>(share_idx + 1);

            // Evaluate polynomial using Horner's method
            // f(x) = c[0] + x*(c[1] + x*(c[2] + ... + x*c[k-1]))
            // Process from highest degree to lowest
            uint8_t result = 0;

            // Start with highest degree coefficient (k-2 index in randomness)
            // and work down to degree 1
            for (int64_t coeff_idx = k - 2; coeff_idx >= 0; coeff_idx--) {
                // Get random coefficient for this degree
                uint8_t coeff = static_cast<uint8_t>(
                    randomness[coeff_idx * secret_len + byte_idx]
                );
                // result = coeff + x * result
                result = gf_add(coeff, gf_multiply(x, result));
            }

            // Final step: add the constant term (secret byte)
            // result = secret_byte + x * result
            result = gf_add(secret_byte, gf_multiply(x, result));

            // Store the share value
            shares[share_idx * secret_len + byte_idx] = static_cast<scalar_t>(result);
        }
    }
}

// Reconstruct secret from k shares using Lagrange interpolation
//
// Given k shares (x_i, y_i) where x_i are the share indices and y_i are
// the share values, reconstructs the original secret by evaluating the
// interpolated polynomial at x = 0.
//
// Parameters:
//   shares:     k shares to use, shape [k, secret_len]
//               Layout: shares[share_idx * secret_len + byte_idx]
//   indices:    x-coordinates of shares (1-indexed share numbers)
//               Length k, values must be distinct and in range [1, 255]
//   output:     Reconstructed secret, length = secret_len
//   secret_len: Length of each share (and output) in bytes
//   k:          Number of shares provided
//
// Note: The indices array contains the original x-values (share numbers)
// that were used during splitting. For example, if shares 1, 3, and 5
// are provided, indices = [1, 3, 5].
template <typename scalar_t>
void shamir_reconstruct_kernel(
    const scalar_t* shares,
    const scalar_t* indices,
    scalar_t* output,
    int64_t secret_len,
    int64_t k
) {
    // Convert indices to uint8_t array for GF operations
    // We use a small stack-allocated buffer for typical cases
    // For very large k, this might need dynamic allocation
    constexpr int64_t MAX_STACK_K = 256;
    uint8_t indices_u8[MAX_STACK_K];

    for (int64_t i = 0; i < k && i < MAX_STACK_K; i++) {
        indices_u8[i] = static_cast<uint8_t>(indices[i]);
    }

    // For each byte position
    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        // Collect share values for this byte position
        uint8_t shares_u8[MAX_STACK_K];
        for (int64_t i = 0; i < k && i < MAX_STACK_K; i++) {
            shares_u8[i] = static_cast<uint8_t>(shares[i * secret_len + byte_idx]);
        }

        // Use Lagrange interpolation to recover the secret (value at x = 0)
        uint8_t secret_byte = gf_lagrange_interpolate(
            shares_u8,
            indices_u8,
            static_cast<int>(k)
        );

        output[byte_idx] = static_cast<scalar_t>(secret_byte);
    }
}

// Verify a share against the secret by re-computing what it should be
//
// This is useful for detecting corrupted or tampered shares before
// attempting reconstruction.
//
// Parameters:
//   share:      The share to verify, length = secret_len
//   share_idx:  The 1-indexed share number (x-coordinate)
//   secret:     The original secret
//   randomness: The random coefficients used during splitting
//   secret_len: Length of secret/share in bytes
//   k:          Threshold value
//
// Returns: true if share is valid, false otherwise
template <typename scalar_t>
bool shamir_verify_share_kernel(
    const scalar_t* share,
    int64_t share_idx,
    const scalar_t* secret,
    const scalar_t* randomness,
    int64_t secret_len,
    int64_t k
) {
    uint8_t x = static_cast<uint8_t>(share_idx);

    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        uint8_t secret_byte = static_cast<uint8_t>(secret[byte_idx]);

        // Compute expected share value using Horner's method
        uint8_t expected = 0;
        for (int64_t coeff_idx = k - 2; coeff_idx >= 0; coeff_idx--) {
            uint8_t coeff = static_cast<uint8_t>(
                randomness[coeff_idx * secret_len + byte_idx]
            );
            expected = gf_add(coeff, gf_multiply(x, expected));
        }
        expected = gf_add(secret_byte, gf_multiply(x, expected));

        // Compare with actual share value
        if (static_cast<uint8_t>(share[byte_idx]) != expected) {
            return false;
        }
    }

    return true;
}

// Refresh shares without changing the secret
//
// This re-randomizes the shares using a new set of random coefficients
// while preserving the same secret. This is useful for proactive secret
// sharing schemes where shares are periodically refreshed to prevent
// attackers from slowly accumulating shares over time.
//
// Parameters:
//   old_shares:     Existing n shares, shape [n, secret_len]
//   new_shares:     Output for refreshed shares, shape [n, secret_len]
//   delta_coeffs:   Random coefficients for a zero-secret polynomial
//                   Size = (k-1) * secret_len
//   secret_len:     Length of each share in bytes
//   n:              Number of shares
//   k:              Threshold
//
// Algorithm: Add evaluations of a random polynomial with zero constant term
// to each existing share. The new shares will reconstruct to the same secret.
template <typename scalar_t>
void shamir_refresh_shares_kernel(
    const scalar_t* old_shares,
    scalar_t* new_shares,
    const scalar_t* delta_coeffs,
    int64_t secret_len,
    int64_t n,
    int64_t k
) {
    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        for (int64_t share_idx = 0; share_idx < n; share_idx++) {
            uint8_t x = static_cast<uint8_t>(share_idx + 1);

            // Evaluate the zero-constant polynomial at x
            // g(x) = d[0]*x + d[1]*x^2 + ... + d[k-2]*x^(k-1)
            //      = x * (d[0] + x * (d[1] + ... + x * d[k-2]))
            uint8_t delta = 0;
            for (int64_t coeff_idx = k - 2; coeff_idx >= 0; coeff_idx--) {
                uint8_t coeff = static_cast<uint8_t>(
                    delta_coeffs[coeff_idx * secret_len + byte_idx]
                );
                delta = gf_add(coeff, gf_multiply(x, delta));
            }
            delta = gf_multiply(x, delta);

            // New share = old share + delta
            uint8_t old_val = static_cast<uint8_t>(
                old_shares[share_idx * secret_len + byte_idx]
            );
            new_shares[share_idx * secret_len + byte_idx] = static_cast<scalar_t>(
                gf_add(old_val, delta)
            );
        }
    }
}

}  // namespace torchscience::kernel::encryption
