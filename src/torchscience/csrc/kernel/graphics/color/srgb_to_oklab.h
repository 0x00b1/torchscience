#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to Oklab.
 *
 * Implements sRGB to Oklab conversion following these steps:
 * 1. Linearize sRGB using the standard transfer function (IEC 61966-2-1:1999)
 * 2. Apply M1 matrix to get LMS
 * 3. Apply cube root to get L'M'S'
 * 4. Apply M2 matrix to get Lab
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param lab Output array [L, a, b] in Oklab
 */
template <typename T>
void srgb_to_oklab_scalar(const T* rgb, T* lab) {
    // sRGB linearization constants
    const T threshold_srgb = T(0.04045);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T gamma = T(2.4);

    // Step 1: Linearize sRGB
    T linear[3];
    for (int i = 0; i < 3; ++i) {
        const T v = rgb[i];
        if (v <= threshold_srgb) {
            linear[i] = v / linear_slope;
        } else {
            linear[i] = std::pow((v + offset) / scale, gamma);
        }
    }

    const T r = linear[0];
    const T g = linear[1];
    const T b = linear[2];

    // Step 2: Apply M1 matrix (linear RGB to LMS)
    const T l = T(0.4122214708) * r + T(0.5363325363) * g + T(0.0514459929) * b;
    const T m = T(0.2119034982) * r + T(0.6806995451) * g + T(0.1073969566) * b;
    const T s = T(0.0883024619) * r + T(0.2817188376) * g + T(0.6299787005) * b;

    // Step 3: Apply cube root (handles negative values correctly)
    const T l_prime = std::cbrt(l);
    const T m_prime = std::cbrt(m);
    const T s_prime = std::cbrt(s);

    // Step 4: Apply M2 matrix (L'M'S' to Oklab)
    lab[0] = T(0.2104542553) * l_prime + T(0.7936177850) * m_prime - T(0.0040720468) * s_prime;
    lab[1] = T(1.9779984951) * l_prime - T(2.4285922050) * m_prime + T(0.4505937099) * s_prime;
    lab[2] = T(0.0259040371) * l_prime + T(0.7827717662) * m_prime - T(0.8086757660) * s_prime;
}

}  // namespace torchscience::kernel::graphics::color
