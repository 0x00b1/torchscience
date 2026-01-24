// src/torchscience/csrc/kernel/encryption/additive.h
//
// Additive (XOR-based) secret sharing scheme implementation.
//
// This implements an n-of-n secret sharing scheme where:
// - A secret is split into n shares
// - ALL n shares are required to reconstruct the secret
// - Fewer than n shares reveal no information about the secret
//
// The scheme uses XOR operations:
// - Split: Generate n-1 random shares, compute last share as secret XOR all random shares
// - Reconstruct: XOR all n shares together to recover the secret

#pragma once

#include <cstdint>

namespace torchscience::kernel::encryption {

// Split secret into n shares using XOR-based additive secret sharing
//
// For each byte of the secret:
//   share[0..n-2] = random bytes
//   share[n-1] = secret XOR share[0] XOR share[1] XOR ... XOR share[n-2]
//
// This ensures that XORing all shares recovers the secret:
//   share[0] XOR share[1] XOR ... XOR share[n-1] = secret
//
// Parameters:
//   secret:     Input secret bytes, length = secret_len
//   shares:     Output array of n shares, each of length secret_len
//               Layout: shares[share_idx * secret_len + byte_idx]
//   randomness: Random bytes for first n-1 shares, size = (n-1) * secret_len
//               Layout: randomness[share_idx * secret_len + byte_idx]
//   secret_len: Length of the secret in bytes
//   n:          Number of shares to generate (must be >= 2)
//
// Security: This is an information-theoretically secure n-of-n scheme.
// Any n-1 shares reveal no information about the secret.
template <typename scalar_t>
void additive_split_kernel(
    const scalar_t* secret,
    scalar_t* shares,
    const scalar_t* randomness,
    int64_t secret_len,
    int64_t n
) {
    // For each byte position in the secret
    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        uint8_t secret_byte = static_cast<uint8_t>(secret[byte_idx]);

        // Accumulate XOR of all random shares
        uint8_t xor_sum = 0;

        // Copy first n-1 shares from randomness
        for (int64_t share_idx = 0; share_idx < n - 1; share_idx++) {
            uint8_t random_byte = static_cast<uint8_t>(
                randomness[share_idx * secret_len + byte_idx]
            );
            shares[share_idx * secret_len + byte_idx] = static_cast<scalar_t>(random_byte);
            xor_sum ^= random_byte;
        }

        // Last share = secret XOR (all random shares XORed together)
        // This ensures: share[0] XOR share[1] XOR ... XOR share[n-1] = secret
        uint8_t last_share = secret_byte ^ xor_sum;
        shares[(n - 1) * secret_len + byte_idx] = static_cast<scalar_t>(last_share);
    }
}

// Reconstruct secret from all n shares using XOR
//
// Computes: output = share[0] XOR share[1] XOR ... XOR share[n-1]
//
// Parameters:
//   shares:     All n shares, layout: shares[share_idx * secret_len + byte_idx]
//   output:     Reconstructed secret, length = secret_len
//   secret_len: Length of each share (and output) in bytes
//   n:          Number of shares
//
// Note: ALL shares must be provided. If any share is missing or incorrect,
// the reconstruction will fail silently (produce incorrect output).
template <typename scalar_t>
void additive_reconstruct_kernel(
    const scalar_t* shares,
    scalar_t* output,
    int64_t secret_len,
    int64_t n
) {
    // For each byte position
    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        // XOR all shares together
        uint8_t result = 0;

        for (int64_t share_idx = 0; share_idx < n; share_idx++) {
            result ^= static_cast<uint8_t>(shares[share_idx * secret_len + byte_idx]);
        }

        output[byte_idx] = static_cast<scalar_t>(result);
    }
}

// Verify that shares reconstruct to expected secret
//
// Parameters:
//   shares:     All n shares
//   secret:     Expected secret
//   secret_len: Length in bytes
//   n:          Number of shares
//
// Returns: true if shares XOR to the expected secret, false otherwise
template <typename scalar_t>
bool additive_verify_kernel(
    const scalar_t* shares,
    const scalar_t* secret,
    int64_t secret_len,
    int64_t n
) {
    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        uint8_t result = 0;

        for (int64_t share_idx = 0; share_idx < n; share_idx++) {
            result ^= static_cast<uint8_t>(shares[share_idx * secret_len + byte_idx]);
        }

        if (result != static_cast<uint8_t>(secret[byte_idx])) {
            return false;
        }
    }

    return true;
}

// Refresh shares without changing the secret
//
// This re-randomizes the shares while preserving the same secret.
// Useful for proactive secret sharing where shares are periodically
// refreshed to prevent attackers from accumulating shares over time.
//
// Parameters:
//   old_shares:     Existing n shares, shape [n, secret_len]
//   new_shares:     Output for refreshed shares, shape [n, secret_len]
//   delta:          Random delta values, shape [n-1, secret_len]
//                   These are XORed into the first n-1 shares
//   secret_len:     Length of each share in bytes
//   n:              Number of shares
//
// Algorithm: For the first n-1 shares, XOR with random delta values.
// The last share is adjusted to maintain the invariant that all shares
// XOR to the same secret.
//
// new_share[i] = old_share[i] XOR delta[i]  for i < n-1
// new_share[n-1] = old_share[n-1] XOR (delta[0] XOR delta[1] XOR ... XOR delta[n-2])
template <typename scalar_t>
void additive_refresh_shares_kernel(
    const scalar_t* old_shares,
    scalar_t* new_shares,
    const scalar_t* delta,
    int64_t secret_len,
    int64_t n
) {
    for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
        // Compute XOR of all delta values for this byte
        uint8_t delta_sum = 0;

        // Apply delta to first n-1 shares and accumulate delta_sum
        for (int64_t share_idx = 0; share_idx < n - 1; share_idx++) {
            uint8_t delta_byte = static_cast<uint8_t>(
                delta[share_idx * secret_len + byte_idx]
            );
            uint8_t old_byte = static_cast<uint8_t>(
                old_shares[share_idx * secret_len + byte_idx]
            );

            new_shares[share_idx * secret_len + byte_idx] = static_cast<scalar_t>(
                old_byte ^ delta_byte
            );
            delta_sum ^= delta_byte;
        }

        // Last share: XOR with sum of all deltas to preserve the secret
        uint8_t old_last = static_cast<uint8_t>(
            old_shares[(n - 1) * secret_len + byte_idx]
        );
        new_shares[(n - 1) * secret_len + byte_idx] = static_cast<scalar_t>(
            old_last ^ delta_sum
        );
    }
}

// Add two sets of additive shares (homomorphic addition)
//
// Additive secret sharing supports homomorphic addition:
// If shares_a encodes secret_a and shares_b encodes secret_b,
// then (shares_a XOR shares_b) encodes (secret_a XOR secret_b).
//
// Parameters:
//   shares_a:   First set of n shares
//   shares_b:   Second set of n shares
//   result:     Output shares encoding secret_a XOR secret_b
//   secret_len: Length of each share in bytes
//   n:          Number of shares
template <typename scalar_t>
void additive_add_shares_kernel(
    const scalar_t* shares_a,
    const scalar_t* shares_b,
    scalar_t* result,
    int64_t secret_len,
    int64_t n
) {
    for (int64_t share_idx = 0; share_idx < n; share_idx++) {
        for (int64_t byte_idx = 0; byte_idx < secret_len; byte_idx++) {
            int64_t idx = share_idx * secret_len + byte_idx;
            result[idx] = static_cast<scalar_t>(
                static_cast<uint8_t>(shares_a[idx]) ^
                static_cast<uint8_t>(shares_b[idx])
            );
        }
    }
}

}  // namespace torchscience::kernel::encryption
