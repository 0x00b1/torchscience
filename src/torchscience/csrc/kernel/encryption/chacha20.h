// src/torchscience/csrc/kernel/encryption/chacha20.h
#pragma once

#include <cstdint>
#include <array>

namespace torchscience::kernel::encryption {

// ChaCha20 quarter round
inline void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    a += b; d ^= a; d = (d << 8) | (d >> 24);
    c += d; b ^= c; b = (b << 7) | (b >> 25);
}

// ChaCha20 block function
inline void chacha20_block(
    std::array<uint32_t, 16>& state,
    const std::array<uint32_t, 16>& input
) {
    state = input;

    // 20 rounds (10 double rounds)
    for (int i = 0; i < 10; i++) {
        // Column rounds
        quarter_round(state[0], state[4], state[8],  state[12]);
        quarter_round(state[1], state[5], state[9],  state[13]);
        quarter_round(state[2], state[6], state[10], state[14]);
        quarter_round(state[3], state[7], state[11], state[15]);
        // Diagonal rounds
        quarter_round(state[0], state[5], state[10], state[15]);
        quarter_round(state[1], state[6], state[11], state[12]);
        quarter_round(state[2], state[7], state[8],  state[13]);
        quarter_round(state[3], state[4], state[9],  state[14]);
    }

    // Add input to state
    for (int i = 0; i < 16; i++) {
        state[i] += input[i];
    }
}

// Load 32-bit little-endian word from bytes
inline uint32_t load_le32(const uint8_t* bytes) {
    return static_cast<uint32_t>(bytes[0])
         | (static_cast<uint32_t>(bytes[1]) << 8)
         | (static_cast<uint32_t>(bytes[2]) << 16)
         | (static_cast<uint32_t>(bytes[3]) << 24);
}

// Store 32-bit word as little-endian bytes
inline void store_le32(uint8_t* bytes, uint32_t word) {
    bytes[0] = static_cast<uint8_t>(word);
    bytes[1] = static_cast<uint8_t>(word >> 8);
    bytes[2] = static_cast<uint8_t>(word >> 16);
    bytes[3] = static_cast<uint8_t>(word >> 24);
}

// Initialize ChaCha20 state from key, nonce, counter
inline void chacha20_init(
    std::array<uint32_t, 16>& state,
    const uint8_t* key,
    const uint8_t* nonce,
    uint32_t counter
) {
    // Constants: "expand 32-byte k"
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;

    // Key (8 words)
    for (int i = 0; i < 8; i++) {
        state[4 + i] = load_le32(key + 4 * i);
    }

    // Counter
    state[12] = counter;

    // Nonce (3 words)
    for (int i = 0; i < 3; i++) {
        state[13 + i] = load_le32(nonce + 4 * i);
    }
}

// Generate keystream bytes
inline void chacha20_keystream(
    uint8_t* output,
    int64_t num_bytes,
    const uint8_t* key,
    const uint8_t* nonce,
    uint32_t counter
) {
    std::array<uint32_t, 16> input_state;
    std::array<uint32_t, 16> output_state;

    chacha20_init(input_state, key, nonce, counter);

    int64_t offset = 0;
    while (offset < num_bytes) {
        chacha20_block(output_state, input_state);

        // Copy output bytes
        int64_t block_bytes = std::min(static_cast<int64_t>(64), num_bytes - offset);
        for (int64_t i = 0; i < block_bytes; i++) {
            output[offset + i] = reinterpret_cast<uint8_t*>(output_state.data())[i];
        }

        offset += 64;
        input_state[12]++;  // Increment counter
    }
}

}  // namespace torchscience::kernel::encryption
