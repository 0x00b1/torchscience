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

inline uint64_t keccak_rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

inline void keccak_f1600(std::array<uint64_t, 25>& st) {
    // Rotation offsets
    constexpr int rotations[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
        27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
    };
    // Pi lane indices
    constexpr int piln[24] = {
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
    };

    for (int round = 0; round < 24; round++) {
        // Theta
        uint64_t bc[5];
        for (int i = 0; i < 5; i++) {
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        }
        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i + 4) % 5] ^ keccak_rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                st[j + i] ^= t;
            }
        }

        // Rho Pi
        uint64_t t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = piln[i];
            uint64_t tmp = st[j];
            st[j] = keccak_rotl64(t, rotations[i]);
            t = tmp;
        }

        // Chi
        for (int j = 0; j < 25; j += 5) {
            uint64_t tmp[5];
            for (int i = 0; i < 5; i++) {
                tmp[i] = st[j + i];
            }
            for (int i = 0; i < 5; i++) {
                st[j + i] = tmp[i] ^ ((~tmp[(i + 1) % 5]) & tmp[(i + 2) % 5]);
            }
        }

        // Iota
        st[0] ^= KECCAK_RC[round];
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
