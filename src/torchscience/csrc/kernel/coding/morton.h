#pragma once

#include <c10/macros/Macros.h>
#include <cstdint>

namespace torchscience::kernel::coding {

// Magic numbers for bit interleaving (3D)
// Spreads bits of x into every 3rd bit position
// CRITICAL: Uses uint64_t throughout to avoid undefined behavior with signed shifts
C10_HOST_DEVICE inline uint64_t spread_bits_3d(uint64_t x) {
    x = (x | (x << 32)) & 0x1f00000000ffffULL;
    x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
    x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x << 2))  & 0x1249249249249249ULL;
    return x;
}

// Compact bits from every 3rd position
// CRITICAL: Uses uint64_t throughout to avoid undefined behavior with signed shifts
C10_HOST_DEVICE inline uint64_t compact_bits_3d(uint64_t x) {
    x = x & 0x1249249249249249ULL;
    x = (x | (x >> 2))  & 0x10c30c30c30c30c3ULL;
    x = (x | (x >> 4))  & 0x100f00f00f00f00fULL;
    x = (x | (x >> 8))  & 0x1f0000ff0000ffULL;
    x = (x | (x >> 16)) & 0x1f00000000ffffULL;
    x = (x | (x >> 32)) & 0x1fffffULL;
    return x;
}

// Magic numbers for bit interleaving (2D)
// Spreads bits of x into every 2nd bit position
C10_HOST_DEVICE inline uint64_t spread_bits_2d(uint64_t x) {
    x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
    x = (x | (x << 8))  & 0x00ff00ff00ff00ffULL;
    x = (x | (x << 4))  & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | (x << 2))  & 0x3333333333333333ULL;
    x = (x | (x << 1))  & 0x5555555555555555ULL;
    return x;
}

// Compact bits from every 2nd position
C10_HOST_DEVICE inline uint64_t compact_bits_2d(uint64_t x) {
    x = x & 0x5555555555555555ULL;
    x = (x | (x >> 1))  & 0x3333333333333333ULL;
    x = (x | (x >> 2))  & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | (x >> 4))  & 0x00ff00ff00ff00ffULL;
    x = (x | (x >> 8))  & 0x0000ffff0000ffffULL;
    x = (x | (x >> 16)) & 0x00000000ffffffffULL;
    return x;
}

// Encode 3D coordinates to Morton code
// Inputs are int64_t for API compatibility, internally uses uint64_t
//
// RANGE REQUIREMENTS:
// - x, y, z must be non-negative
// - For raw Morton: x, y, z < 2^21 (fits in 63 bits interleaved)
// Debug builds should assert these bounds.
//
C10_HOST_DEVICE inline int64_t morton_encode_3d(int64_t x, int64_t y, int64_t z) {
    uint64_t ux = static_cast<uint64_t>(x);
    uint64_t uy = static_cast<uint64_t>(y);
    uint64_t uz = static_cast<uint64_t>(z);
    uint64_t result = spread_bits_3d(ux) | (spread_bits_3d(uy) << 1) | (spread_bits_3d(uz) << 2);
    return static_cast<int64_t>(result);
}

// Decode Morton code to 3D coordinates
// Input is int64_t for API compatibility, internally uses uint64_t
C10_HOST_DEVICE inline void morton_decode_3d(int64_t code, int64_t& x, int64_t& y, int64_t& z) {
    uint64_t ucode = static_cast<uint64_t>(code);
    x = static_cast<int64_t>(compact_bits_3d(ucode));
    y = static_cast<int64_t>(compact_bits_3d(ucode >> 1));
    z = static_cast<int64_t>(compact_bits_3d(ucode >> 2));
}

// Encode 2D coordinates to Morton code
C10_HOST_DEVICE inline int64_t morton_encode_2d(int64_t x, int64_t y) {
    uint64_t ux = static_cast<uint64_t>(x);
    uint64_t uy = static_cast<uint64_t>(y);
    uint64_t result = spread_bits_2d(ux) | (spread_bits_2d(uy) << 1);
    return static_cast<int64_t>(result);
}

// Decode Morton code to 2D coordinates
C10_HOST_DEVICE inline void morton_decode_2d(int64_t code, int64_t& x, int64_t& y) {
    uint64_t ucode = static_cast<uint64_t>(code);
    x = static_cast<int64_t>(compact_bits_2d(ucode));
    y = static_cast<int64_t>(compact_bits_2d(ucode >> 1));
}

// Encode octree code with depth (3D only)
// Uses uint64_t internally to avoid signed overflow UB
// SAFETY: Masks morton to 60 bits to prevent overflow into depth field
//
// Bit layout:
// Bits 63-60: depth level (0-15, supporting 2^15 = 32K resolution per axis)
// Bits 59-0:  interleaved x, y, z coordinates (20 bits each)
//
C10_HOST_DEVICE inline int64_t octree_encode(int64_t depth, int64_t x, int64_t y, int64_t z) {
    uint64_t udepth = static_cast<uint64_t>(depth);
    uint64_t umorton = static_cast<uint64_t>(morton_encode_3d(x, y, z));
    // Mask to 60 bits - coordinates must be < 2^20 each (enforced by depth <= 15)
    umorton &= 0x0FFFFFFFFFFFFFFFULL;
    return static_cast<int64_t>((udepth << 60) | umorton);
}

// Decode octree code
// CRITICAL: Cast to uint64_t before bit operations to handle depth >= 8
C10_HOST_DEVICE inline void octree_decode(int64_t code, int64_t& depth, int64_t& x, int64_t& y, int64_t& z) {
    uint64_t ucode = static_cast<uint64_t>(code);
    depth = static_cast<int64_t>((ucode >> 60) & 0xFULL);
    morton_decode_3d(static_cast<int64_t>(ucode & 0x0FFFFFFFFFFFFFFFULL), x, y, z);
}

// Get parent code (one level up)
// Uses uint64_t to avoid UB with signed shifts
C10_HOST_DEVICE inline int64_t octree_parent(int64_t code) {
    uint64_t ucode = static_cast<uint64_t>(code);
    uint64_t depth = (ucode >> 60) & 0xFULL;
    if (depth == 0) return code;  // Root has no parent
    uint64_t morton = ucode & 0x0FFFFFFFFFFFFFFFULL;
    // Shift morton code right by 3 bits (remove lowest x,y,z bits)
    uint64_t parent_morton = morton >> 3;
    return static_cast<int64_t>(((depth - 1) << 60) | parent_morton);
}

// Get child code for octant (0-7)
// Uses uint64_t to avoid UB with signed shifts
C10_HOST_DEVICE inline int64_t octree_child(int64_t code, int64_t octant) {
    uint64_t ucode = static_cast<uint64_t>(code);
    uint64_t depth = (ucode >> 60) & 0xFULL;
    uint64_t morton = ucode & 0x0FFFFFFFFFFFFFFFULL;
    uint64_t child_morton = (morton << 3) | static_cast<uint64_t>(octant);
    return static_cast<int64_t>(((depth + 1) << 60) | child_morton);
}

// Get octant index (0-7) that this code occupies within its parent
C10_HOST_DEVICE inline int64_t octree_octant(int64_t code) {
    uint64_t ucode = static_cast<uint64_t>(code);
    uint64_t morton = ucode & 0x0FFFFFFFFFFFFFFFULL;
    return static_cast<int64_t>(morton & 0x7ULL);
}

}  // namespace torchscience::kernel::coding
