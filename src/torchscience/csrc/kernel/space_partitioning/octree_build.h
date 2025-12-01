#pragma once

#include <c10/macros/Macros.h>
#include <cstdint>
#include <algorithm>

#include "../coding/morton.h"

namespace torchscience::kernel::space_partitioning {

// Aggregation mode enum (must match Python constants)
enum class AggregationMode : int64_t {
    MEAN = 0,
    SUM = 1,
    MAX = 2,
};

// Maximum linear probes for hash table lookup
constexpr int64_t MAX_PROBES = 64;

// Hash function for Morton codes using MurmurHash3 finalizer
// Uses uint64_t throughout to avoid signed overflow UB in C++
C10_HOST_DEVICE inline int64_t hash_code(int64_t code, int64_t capacity) {
    uint64_t h = static_cast<uint64_t>(code);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    // Assumes power-of-2 capacity
    return static_cast<int64_t>(h & static_cast<uint64_t>(capacity - 1));
}

// Quantize [-1, 1] coordinate to integer at given depth
// Always uses fp32 internally to avoid precision loss with fp16
C10_HOST_DEVICE inline int64_t quantize(float x, int64_t depth) {
    float scale = static_cast<float>(1LL << depth);
    float normalized = (x + 1.0f) * 0.5f;  // [0, 1]
    // Clamp to [0, 1] (silent clamp for out-of-bounds)
    if (normalized < 0.0f) normalized = 0.0f;
    if (normalized > 1.0f) normalized = 1.0f;
    int64_t result = static_cast<int64_t>(normalized * scale);
    // Clamp to valid range
    if (result >= (1LL << depth)) result = (1LL << depth) - 1;
    return result;
}

// Compute octree Morton code from point in [-1, 1]³
C10_HOST_DEVICE inline int64_t point_to_code(float x, float y, float z, int64_t depth) {
    int64_t ix = quantize(x, depth);
    int64_t iy = quantize(y, depth);
    int64_t iz = quantize(z, depth);
    return coding::octree_encode(depth, ix, iy, iz);
}

// Hash table lookup with bounded linear probing
// Returns index if found, -1 if not found
C10_HOST_DEVICE inline int64_t hash_lookup(
    const int64_t* structure,
    const int64_t* codes,
    int64_t capacity,
    int64_t query_code
) {
    int64_t slot = hash_code(query_code, capacity);
    for (int64_t i = 0; i < MAX_PROBES; ++i) {
        int64_t idx = structure[(slot + i) & (capacity - 1)];
        if (idx == -1) return -1;  // Empty slot, not found
        if (codes[idx] == query_code) return idx;
    }
    return -1;  // Exceeded max probes
}

// Hash table insert with linear probing
// Returns displacement (number of probes needed)
// Returns -1 if insertion failed (no empty slot within MAX_PROBES)
C10_HOST_DEVICE inline int64_t hash_insert(
    int64_t* structure,
    const int64_t* codes,
    int64_t capacity,
    int64_t code,
    int64_t index
) {
    int64_t slot = hash_code(code, capacity);
    for (int64_t i = 0; i < MAX_PROBES; ++i) {
        int64_t target_slot = (slot + i) & (capacity - 1);
        if (structure[target_slot] == -1) {
            structure[target_slot] = index;
            return i;  // Return displacement
        }
    }
    return -1;  // Failed to insert
}

// Round up to next power of 2
C10_HOST_DEVICE inline int64_t next_power_of_2(int64_t n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

// Extract depth from octree code with proper masking
C10_HOST_DEVICE inline int64_t get_depth(int64_t code) {
    uint64_t ucode = static_cast<uint64_t>(code);
    return static_cast<int64_t>((ucode >> 60) & 0xFULL);
}

// Compute octant index (0-7) for a point at given depth
C10_HOST_DEVICE inline int64_t compute_octant(float x, float y, float z, int64_t depth) {
    int64_t ix = quantize(x, depth);
    int64_t iy = quantize(y, depth);
    int64_t iz = quantize(z, depth);
    return (ix & 1) | ((iy & 1) << 1) | ((iz & 1) << 2);
}

}  // namespace torchscience::kernel::space_partitioning
