#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single Oklab pixel to sRGB.
 *
 * Implements Oklab to sRGB conversion following these steps:
 * 1. Apply inverse M2 matrix to get L'M'S'
 * 2. Cube to get LMS
 * 3. Apply inverse M1 matrix to get linear RGB
 * 4. Apply sRGB gamma encoding
 *
 * @param lab Input array [L, a, b] in Oklab
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void oklab_to_srgb_scalar(const T* lab, T* rgb) {
    // sRGB encoding constants
    const T threshold_linear = T(0.0031308);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    const T L = lab[0];
    const T a = lab[1];
    const T b = lab[2];

    // Step 1: Apply inverse M2 matrix (Oklab to L'M'S')
    const T l_prime = L + T(0.3963377774) * a + T(0.2158037573) * b;
    const T m_prime = L - T(0.1055613458) * a - T(0.0638541728) * b;
    const T s_prime = L - T(0.0894841775) * a - T(1.2914855480) * b;

    // Step 2: Cube to get LMS
    const T l = l_prime * l_prime * l_prime;
    const T m = m_prime * m_prime * m_prime;
    const T s = s_prime * s_prime * s_prime;

    // Step 3: Apply inverse M1 matrix (LMS to linear RGB)
    const T r_linear = T(+4.0767416621) * l - T(3.3077115913) * m + T(0.2309699292) * s;
    const T g_linear = T(-1.2684380046) * l + T(2.6097574011) * m - T(0.3413193965) * s;
    const T b_linear = T(-0.0041960863) * l - T(0.7034186147) * m + T(1.7076147010) * s;

    // Step 4: Apply sRGB gamma encoding
    T linear[3] = {r_linear, g_linear, b_linear};
    for (int i = 0; i < 3; ++i) {
        const T v = linear[i];
        if (v <= threshold_linear) {
            rgb[i] = v * linear_slope;
        } else if (v > T(0)) {
            rgb[i] = scale * std::pow(v, inv_gamma) - offset;
        } else {
            // Handle negative values (out of gamut)
            rgb[i] = -scale * std::pow(-v, inv_gamma) + offset;
        }
    }
}

}  // namespace torchscience::kernel::graphics::color
