#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single CIE XYZ pixel to sRGB.
 *
 * Implements IEC 61966-2-1:1999 XYZ to sRGB conversion:
 * 1. Apply the XYZ to RGB matrix for D65 illuminant
 * 2. Apply gamma encoding (companding)
 *
 * @param xyz Input array [X, Y, Z]
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void xyz_to_srgb_scalar(const T* xyz, T* rgb) {
    const T X = xyz[0];
    const T Y = xyz[1];
    const T Z = xyz[2];

    // Apply XYZ to linear RGB matrix (inverse of sRGB D65)
    const T r_linear = T( 3.2404542) * X + T(-1.5371385) * Y + T(-0.4985314) * Z;
    const T g_linear = T(-0.9692660) * X + T( 1.8760108) * Y + T( 0.0415560) * Z;
    const T b_linear = T( 0.0556434) * X + T(-0.2040259) * Y + T( 1.0572252) * Z;

    // Gamma encode (linear to sRGB)
    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    T linear[3] = {r_linear, g_linear, b_linear};
    for (int i = 0; i < 3; ++i) {
        const T value = linear[i];
        if (value <= threshold) {
            rgb[i] = linear_slope * value;
        } else {
            rgb[i] = scale * std::pow(value, inv_gamma) - offset;
        }
    }
}

}  // namespace torchscience::kernel::graphics::color
