// src/torchscience/csrc/kernel/encryption/aes.h
#pragma once

#include <cstdint>
#include <array>
#include <cstring>

namespace torchscience::kernel::encryption {

// AES S-box (SubBytes transformation lookup table)
constexpr std::array<uint8_t, 256> AES_SBOX = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// AES Inverse S-box (InvSubBytes transformation lookup table)
constexpr std::array<uint8_t, 256> AES_INV_SBOX = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};

// Round constants for key expansion
// Rcon[i] contains the values given by x^(i-1) in GF(2^8)
constexpr std::array<uint8_t, 11> AES_RCON = {
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};

// AES block size in bytes
constexpr int AES_BLOCK_SIZE = 16;

// Number of rounds for different key sizes
constexpr int AES_128_ROUNDS = 10;
constexpr int AES_256_ROUNDS = 14;

// Key sizes in bytes
constexpr int AES_128_KEY_SIZE = 16;
constexpr int AES_256_KEY_SIZE = 32;

// Expanded key sizes in bytes
constexpr int AES_128_EXPANDED_KEY_SIZE = 176;  // 16 * (10 + 1)
constexpr int AES_256_EXPANDED_KEY_SIZE = 240;  // 16 * (14 + 1)

// GF(2^8) multiplication by 2 (xtime operation)
// Used in MixColumns transformation
inline uint8_t xtime(uint8_t x) {
    return static_cast<uint8_t>((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

// GF(2^8) multiplication
// Multiply two bytes in the Galois Field GF(2^8)
inline uint8_t gf_mul(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    uint8_t hi_bit_set;
    for (int i = 0; i < 8; i++) {
        if (b & 1) {
            result ^= a;
        }
        hi_bit_set = a & 0x80;
        a <<= 1;
        if (hi_bit_set) {
            a ^= 0x1b;  // Reduction polynomial: x^8 + x^4 + x^3 + x + 1
        }
        b >>= 1;
    }
    return result;
}

// Rotate word left by 1 byte (for key expansion)
inline void rot_word(uint8_t* word) {
    uint8_t temp = word[0];
    word[0] = word[1];
    word[1] = word[2];
    word[2] = word[3];
    word[3] = temp;
}

// Apply S-box to each byte of a word (for key expansion)
inline void sub_word(uint8_t* word) {
    word[0] = AES_SBOX[word[0]];
    word[1] = AES_SBOX[word[1]];
    word[2] = AES_SBOX[word[2]];
    word[3] = AES_SBOX[word[3]];
}

// AES-128 Key Expansion
// Expands 16-byte key to 176 bytes (11 round keys)
inline void aes_key_expansion_128(const uint8_t* key, uint8_t* round_keys) {
    // First round key is the key itself
    std::memcpy(round_keys, key, AES_128_KEY_SIZE);

    uint8_t temp[4];
    int bytes_generated = AES_128_KEY_SIZE;
    int rcon_index = 1;

    while (bytes_generated < AES_128_EXPANDED_KEY_SIZE) {
        // Copy last 4 bytes of previous round key
        temp[0] = round_keys[bytes_generated - 4];
        temp[1] = round_keys[bytes_generated - 3];
        temp[2] = round_keys[bytes_generated - 2];
        temp[3] = round_keys[bytes_generated - 1];

        // Every 16 bytes (4 words), apply transformations
        if (bytes_generated % AES_128_KEY_SIZE == 0) {
            rot_word(temp);
            sub_word(temp);
            temp[0] ^= AES_RCON[rcon_index++];
        }

        // XOR with word 4 positions back
        round_keys[bytes_generated] = round_keys[bytes_generated - AES_128_KEY_SIZE] ^ temp[0];
        round_keys[bytes_generated + 1] = round_keys[bytes_generated - AES_128_KEY_SIZE + 1] ^ temp[1];
        round_keys[bytes_generated + 2] = round_keys[bytes_generated - AES_128_KEY_SIZE + 2] ^ temp[2];
        round_keys[bytes_generated + 3] = round_keys[bytes_generated - AES_128_KEY_SIZE + 3] ^ temp[3];

        bytes_generated += 4;
    }
}

// AES-256 Key Expansion
// Expands 32-byte key to 240 bytes (15 round keys)
inline void aes_key_expansion_256(const uint8_t* key, uint8_t* round_keys) {
    // First two round keys are the key itself
    std::memcpy(round_keys, key, AES_256_KEY_SIZE);

    uint8_t temp[4];
    int bytes_generated = AES_256_KEY_SIZE;
    int rcon_index = 1;

    while (bytes_generated < AES_256_EXPANDED_KEY_SIZE) {
        // Copy last 4 bytes
        temp[0] = round_keys[bytes_generated - 4];
        temp[1] = round_keys[bytes_generated - 3];
        temp[2] = round_keys[bytes_generated - 2];
        temp[3] = round_keys[bytes_generated - 1];

        // Every 32 bytes, apply RotWord, SubWord, and Rcon
        if (bytes_generated % AES_256_KEY_SIZE == 0) {
            rot_word(temp);
            sub_word(temp);
            temp[0] ^= AES_RCON[rcon_index++];
        }
        // Every 16 bytes (but not every 32), apply SubWord only
        else if (bytes_generated % AES_256_KEY_SIZE == 16) {
            sub_word(temp);
        }

        // XOR with word 32 bytes back
        round_keys[bytes_generated] = round_keys[bytes_generated - AES_256_KEY_SIZE] ^ temp[0];
        round_keys[bytes_generated + 1] = round_keys[bytes_generated - AES_256_KEY_SIZE + 1] ^ temp[1];
        round_keys[bytes_generated + 2] = round_keys[bytes_generated - AES_256_KEY_SIZE + 2] ^ temp[2];
        round_keys[bytes_generated + 3] = round_keys[bytes_generated - AES_256_KEY_SIZE + 3] ^ temp[3];

        bytes_generated += 4;
    }
}

// SubBytes transformation
// Substitute each byte using the S-box
inline void aes_sub_bytes(uint8_t* state) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        state[i] = AES_SBOX[state[i]];
    }
}

// InvSubBytes transformation
// Substitute each byte using the inverse S-box
inline void aes_inv_sub_bytes(uint8_t* state) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        state[i] = AES_INV_SBOX[state[i]];
    }
}

// ShiftRows transformation
// State is stored column-major: state[row + 4*col]
// Row 0: no shift
// Row 1: shift left by 1
// Row 2: shift left by 2
// Row 3: shift left by 3
inline void aes_shift_rows(uint8_t* state) {
    uint8_t temp;

    // Row 1: shift left by 1
    temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;

    // Row 2: shift left by 2
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Row 3: shift left by 3 (equivalent to shift right by 1)
    temp = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = state[3];
    state[3] = temp;
}

// InvShiftRows transformation
// Row 0: no shift
// Row 1: shift right by 1
// Row 2: shift right by 2
// Row 3: shift right by 3
inline void aes_inv_shift_rows(uint8_t* state) {
    uint8_t temp;

    // Row 1: shift right by 1
    temp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = temp;

    // Row 2: shift right by 2
    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;

    // Row 3: shift right by 3 (equivalent to shift left by 1)
    temp = state[3];
    state[3] = state[7];
    state[7] = state[11];
    state[11] = state[15];
    state[15] = temp;
}

// MixColumns transformation
// Each column is treated as a polynomial over GF(2^8) and multiplied by
// a(x) = {03}x^3 + {01}x^2 + {01}x + {02} modulo x^4 + 1
inline void aes_mix_columns(uint8_t* state) {
    uint8_t temp[4];
    for (int col = 0; col < 4; col++) {
        int base = col * 4;
        temp[0] = gf_mul(0x02, state[base]) ^ gf_mul(0x03, state[base + 1]) ^
                  state[base + 2] ^ state[base + 3];
        temp[1] = state[base] ^ gf_mul(0x02, state[base + 1]) ^
                  gf_mul(0x03, state[base + 2]) ^ state[base + 3];
        temp[2] = state[base] ^ state[base + 1] ^
                  gf_mul(0x02, state[base + 2]) ^ gf_mul(0x03, state[base + 3]);
        temp[3] = gf_mul(0x03, state[base]) ^ state[base + 1] ^
                  state[base + 2] ^ gf_mul(0x02, state[base + 3]);

        state[base] = temp[0];
        state[base + 1] = temp[1];
        state[base + 2] = temp[2];
        state[base + 3] = temp[3];
    }
}

// InvMixColumns transformation
// Multiply by the inverse polynomial:
// a^-1(x) = {0b}x^3 + {0d}x^2 + {09}x + {0e}
inline void aes_inv_mix_columns(uint8_t* state) {
    uint8_t temp[4];
    for (int col = 0; col < 4; col++) {
        int base = col * 4;
        temp[0] = gf_mul(0x0e, state[base]) ^ gf_mul(0x0b, state[base + 1]) ^
                  gf_mul(0x0d, state[base + 2]) ^ gf_mul(0x09, state[base + 3]);
        temp[1] = gf_mul(0x09, state[base]) ^ gf_mul(0x0e, state[base + 1]) ^
                  gf_mul(0x0b, state[base + 2]) ^ gf_mul(0x0d, state[base + 3]);
        temp[2] = gf_mul(0x0d, state[base]) ^ gf_mul(0x09, state[base + 1]) ^
                  gf_mul(0x0e, state[base + 2]) ^ gf_mul(0x0b, state[base + 3]);
        temp[3] = gf_mul(0x0b, state[base]) ^ gf_mul(0x0d, state[base + 1]) ^
                  gf_mul(0x09, state[base + 2]) ^ gf_mul(0x0e, state[base + 3]);

        state[base] = temp[0];
        state[base + 1] = temp[1];
        state[base + 2] = temp[2];
        state[base + 3] = temp[3];
    }
}

// AddRoundKey transformation
// XOR state with round key
inline void aes_add_round_key(uint8_t* state, const uint8_t* round_key) {
    for (int i = 0; i < AES_BLOCK_SIZE; i++) {
        state[i] ^= round_key[i];
    }
}

// AES block encryption (ECB mode, single block)
// key_len: 16 for AES-128, 32 for AES-256
inline void aes_encrypt_block(
    uint8_t* output,
    const uint8_t* input,
    const uint8_t* key,
    int key_len
) {
    // Determine number of rounds based on key length
    int num_rounds = (key_len == AES_256_KEY_SIZE) ? AES_256_ROUNDS : AES_128_ROUNDS;

    // Expand key
    uint8_t round_keys[AES_256_EXPANDED_KEY_SIZE];
    if (key_len == AES_256_KEY_SIZE) {
        aes_key_expansion_256(key, round_keys);
    } else {
        aes_key_expansion_128(key, round_keys);
    }

    // Copy input to state
    uint8_t state[AES_BLOCK_SIZE];
    std::memcpy(state, input, AES_BLOCK_SIZE);

    // Initial round key addition
    aes_add_round_key(state, round_keys);

    // Main rounds (all but last)
    for (int round = 1; round < num_rounds; round++) {
        aes_sub_bytes(state);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, round_keys + round * AES_BLOCK_SIZE);
    }

    // Final round (no MixColumns)
    aes_sub_bytes(state);
    aes_shift_rows(state);
    aes_add_round_key(state, round_keys + num_rounds * AES_BLOCK_SIZE);

    // Copy state to output
    std::memcpy(output, state, AES_BLOCK_SIZE);
}

// AES block decryption (ECB mode, single block)
// key_len: 16 for AES-128, 32 for AES-256
inline void aes_decrypt_block(
    uint8_t* output,
    const uint8_t* input,
    const uint8_t* key,
    int key_len
) {
    // Determine number of rounds based on key length
    int num_rounds = (key_len == AES_256_KEY_SIZE) ? AES_256_ROUNDS : AES_128_ROUNDS;

    // Expand key
    uint8_t round_keys[AES_256_EXPANDED_KEY_SIZE];
    if (key_len == AES_256_KEY_SIZE) {
        aes_key_expansion_256(key, round_keys);
    } else {
        aes_key_expansion_128(key, round_keys);
    }

    // Copy input to state
    uint8_t state[AES_BLOCK_SIZE];
    std::memcpy(state, input, AES_BLOCK_SIZE);

    // Initial round key addition (using last round key)
    aes_add_round_key(state, round_keys + num_rounds * AES_BLOCK_SIZE);

    // Main rounds (all but last, in reverse)
    for (int round = num_rounds - 1; round > 0; round--) {
        aes_inv_shift_rows(state);
        aes_inv_sub_bytes(state);
        aes_add_round_key(state, round_keys + round * AES_BLOCK_SIZE);
        aes_inv_mix_columns(state);
    }

    // Final round (no InvMixColumns)
    aes_inv_shift_rows(state);
    aes_inv_sub_bytes(state);
    aes_add_round_key(state, round_keys);

    // Copy state to output
    std::memcpy(output, state, AES_BLOCK_SIZE);
}

// AES block encryption with pre-expanded keys
inline void aes_encrypt_block_with_expanded_key(
    uint8_t* output,
    const uint8_t* input,
    const uint8_t* round_keys,
    int num_rounds
) {
    // Copy input to state
    uint8_t state[AES_BLOCK_SIZE];
    std::memcpy(state, input, AES_BLOCK_SIZE);

    // Initial round key addition
    aes_add_round_key(state, round_keys);

    // Main rounds (all but last)
    for (int round = 1; round < num_rounds; round++) {
        aes_sub_bytes(state);
        aes_shift_rows(state);
        aes_mix_columns(state);
        aes_add_round_key(state, round_keys + round * AES_BLOCK_SIZE);
    }

    // Final round (no MixColumns)
    aes_sub_bytes(state);
    aes_shift_rows(state);
    aes_add_round_key(state, round_keys + num_rounds * AES_BLOCK_SIZE);

    // Copy state to output
    std::memcpy(output, state, AES_BLOCK_SIZE);
}

// AES-CTR mode encryption/decryption
// CTR mode is symmetric: encryption and decryption are the same operation
// nonce: 12 bytes
// counter: initial counter value (32-bit, big-endian in counter block)
inline void aes_ctr(
    uint8_t* output,
    const uint8_t* input,
    int64_t len,
    const uint8_t* key,
    int key_len,
    const uint8_t* nonce,
    uint32_t counter
) {
    // Determine number of rounds based on key length
    int num_rounds = (key_len == AES_256_KEY_SIZE) ? AES_256_ROUNDS : AES_128_ROUNDS;

    // Expand key once
    uint8_t round_keys[AES_256_EXPANDED_KEY_SIZE];
    if (key_len == AES_256_KEY_SIZE) {
        aes_key_expansion_256(key, round_keys);
    } else {
        aes_key_expansion_128(key, round_keys);
    }

    // Counter block: nonce (12 bytes) || counter (4 bytes big-endian)
    uint8_t counter_block[AES_BLOCK_SIZE];
    std::memcpy(counter_block, nonce, 12);

    // Keystream block
    uint8_t keystream[AES_BLOCK_SIZE];

    int64_t offset = 0;
    while (offset < len) {
        // Set counter in big-endian format
        counter_block[12] = static_cast<uint8_t>(counter >> 24);
        counter_block[13] = static_cast<uint8_t>(counter >> 16);
        counter_block[14] = static_cast<uint8_t>(counter >> 8);
        counter_block[15] = static_cast<uint8_t>(counter);

        // Encrypt counter block to generate keystream
        aes_encrypt_block_with_expanded_key(keystream, counter_block, round_keys, num_rounds);

        // XOR keystream with input to produce output
        int64_t block_len = std::min(static_cast<int64_t>(AES_BLOCK_SIZE), len - offset);
        for (int64_t i = 0; i < block_len; i++) {
            output[offset + i] = input[offset + i] ^ keystream[i];
        }

        offset += AES_BLOCK_SIZE;
        counter++;
    }
}

// AES-CTR mode with 64-bit counter (for larger data)
// nonce: 8 bytes
// counter: initial counter value (64-bit, big-endian in counter block)
inline void aes_ctr_64(
    uint8_t* output,
    const uint8_t* input,
    int64_t len,
    const uint8_t* key,
    int key_len,
    const uint8_t* nonce,
    uint64_t counter
) {
    // Determine number of rounds based on key length
    int num_rounds = (key_len == AES_256_KEY_SIZE) ? AES_256_ROUNDS : AES_128_ROUNDS;

    // Expand key once
    uint8_t round_keys[AES_256_EXPANDED_KEY_SIZE];
    if (key_len == AES_256_KEY_SIZE) {
        aes_key_expansion_256(key, round_keys);
    } else {
        aes_key_expansion_128(key, round_keys);
    }

    // Counter block: nonce (8 bytes) || counter (8 bytes big-endian)
    uint8_t counter_block[AES_BLOCK_SIZE];
    std::memcpy(counter_block, nonce, 8);

    // Keystream block
    uint8_t keystream[AES_BLOCK_SIZE];

    int64_t offset = 0;
    while (offset < len) {
        // Set counter in big-endian format
        counter_block[8] = static_cast<uint8_t>(counter >> 56);
        counter_block[9] = static_cast<uint8_t>(counter >> 48);
        counter_block[10] = static_cast<uint8_t>(counter >> 40);
        counter_block[11] = static_cast<uint8_t>(counter >> 32);
        counter_block[12] = static_cast<uint8_t>(counter >> 24);
        counter_block[13] = static_cast<uint8_t>(counter >> 16);
        counter_block[14] = static_cast<uint8_t>(counter >> 8);
        counter_block[15] = static_cast<uint8_t>(counter);

        // Encrypt counter block to generate keystream
        aes_encrypt_block_with_expanded_key(keystream, counter_block, round_keys, num_rounds);

        // XOR keystream with input to produce output
        int64_t block_len = std::min(static_cast<int64_t>(AES_BLOCK_SIZE), len - offset);
        for (int64_t i = 0; i < block_len; i++) {
            output[offset + i] = input[offset + i] ^ keystream[i];
        }

        offset += AES_BLOCK_SIZE;
        counter++;
    }
}

}  // namespace torchscience::kernel::encryption
