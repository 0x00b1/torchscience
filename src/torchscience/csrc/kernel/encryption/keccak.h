#pragma once

#include <cstdint>
#include <array>
#include <cstring>

namespace torchscience::kernel::encryption {

// Keccak-f[1600] round constants
constexpr std::array<uint64_t, 24> KECCAK_RC = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets for rho step
constexpr std::array<int, 25> KECCAK_RHO = {
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20,
    3, 10, 43, 25, 39, 41, 45, 15, 21, 8,
    18, 2, 61, 56, 14
};

// Pi step permutation indices
constexpr std::array<int, 25> KECCAK_PI = {
    0, 6, 12, 18, 24, 3, 9, 10, 16, 22,
    1, 7, 13, 19, 20, 4, 5, 11, 17, 23,
    2, 8, 14, 15, 21
};

inline uint64_t keccak_rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

inline void keccak_f1600(std::array<uint64_t, 25>& state) {
    for (int round = 0; round < 24; round++) {
        // Theta step
        std::array<uint64_t, 5> c;
        for (int x = 0; x < 5; x++) {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        std::array<uint64_t, 5> d;
        for (int x = 0; x < 5; x++) {
            d[x] = c[(x + 4) % 5] ^ keccak_rotl64(c[(x + 1) % 5], 1);
        }
        for (int i = 0; i < 25; i++) {
            state[i] ^= d[i % 5];
        }

        // Rho and Pi steps combined
        std::array<uint64_t, 25> temp;
        for (int i = 0; i < 25; i++) {
            temp[KECCAK_PI[i]] = keccak_rotl64(state[i], KECCAK_RHO[i]);
        }

        // Chi step
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                int i = y * 5 + x;
                state[i] = temp[i] ^ ((~temp[y * 5 + (x + 1) % 5]) & temp[y * 5 + (x + 2) % 5]);
            }
        }

        // Iota step
        state[0] ^= KECCAK_RC[round];
    }
}

inline void keccak_hash(
    uint8_t* output,
    const uint8_t* input,
    int64_t input_len,
    int rate,
    int output_len,
    uint8_t domain_sep
) {
    std::array<uint64_t, 25> state = {};

    // Absorb phase
    int64_t offset = 0;
    while (offset + rate <= input_len) {
        for (int i = 0; i < rate / 8; i++) {
            uint64_t lane = 0;
            for (int j = 0; j < 8; j++) {
                lane |= static_cast<uint64_t>(input[offset + i * 8 + j]) << (j * 8);
            }
            state[i] ^= lane;
        }
        keccak_f1600(state);
        offset += rate;
    }

    // Pad and absorb final block
    std::array<uint8_t, 200> last_block = {};
    int64_t remaining = input_len - offset;
    std::memcpy(last_block.data(), input + offset, remaining);
    last_block[remaining] = domain_sep;
    last_block[rate - 1] |= 0x80;

    for (int i = 0; i < rate / 8; i++) {
        uint64_t lane = 0;
        for (int j = 0; j < 8; j++) {
            lane |= static_cast<uint64_t>(last_block[i * 8 + j]) << (j * 8);
        }
        state[i] ^= lane;
    }
    keccak_f1600(state);

    // Squeeze phase
    for (int i = 0; i < output_len; i++) {
        output[i] = static_cast<uint8_t>(state[i / 8] >> ((i % 8) * 8));
    }
}

inline void sha3_256_hash(uint8_t* output, const uint8_t* input, int64_t input_len) {
    keccak_hash(output, input, input_len, 136, 32, 0x06);
}

inline void sha3_512_hash(uint8_t* output, const uint8_t* input, int64_t input_len) {
    keccak_hash(output, input, input_len, 72, 64, 0x06);
}

inline void keccak256_hash(uint8_t* output, const uint8_t* input, int64_t input_len) {
    keccak_hash(output, input, input_len, 136, 32, 0x01);
}

}  // namespace torchscience::kernel::encryption
