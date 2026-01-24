#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to CIE XYZ.
 *
 * Implements IEC 61966-2-1:1999 sRGB to XYZ conversion:
 * 1. Linearize sRGB using the standard transfer function
 * 2. Apply the RGB to XYZ matrix for D65 illuminant
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param xyz Output array [X, Y, Z]
 */
template <typename T>
void srgb_to_xyz_scalar(const T* rgb, T* xyz) {
    const T threshold = T(0.04045);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T gamma = T(2.4);

    // Linearize each channel
    T linear[3];
    for (int i = 0; i < 3; ++i) {
        const T value = rgb[i];
        if (value <= threshold) {
            linear[i] = value / linear_slope;
        } else {
            linear[i] = std::pow((value + offset) / scale, gamma);
        }
    }

    const T r = linear[0];
    const T g = linear[1];
    const T b = linear[2];

    // Apply RGB to XYZ matrix (sRGB D65)
    // Matrix from IEC 61966-2-1:1999
    xyz[0] = T(0.4124564) * r + T(0.3575761) * g + T(0.1804375) * b;  // X
    xyz[1] = T(0.2126729) * r + T(0.7151522) * g + T(0.0721750) * b;  // Y
    xyz[2] = T(0.0193339) * r + T(0.1191920) * g + T(0.9503041) * b;  // Z
}

}  // namespace torchscience::kernel::graphics::color
