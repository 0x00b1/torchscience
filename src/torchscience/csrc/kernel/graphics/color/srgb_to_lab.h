#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to CIELAB.
 *
 * Implements sRGB to CIELAB conversion following these steps:
 * 1. Linearize sRGB using the standard transfer function (IEC 61966-2-1:1999)
 * 2. Apply the RGB to XYZ matrix for D65 illuminant
 * 3. Convert XYZ to Lab using the f(t) helper function
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param lab Output array [L, a, b] where L in [0,100], a,b typically in [-128,127]
 */
template <typename T>
void srgb_to_lab_scalar(const T* rgb, T* lab) {
    // sRGB linearization constants
    const T threshold_srgb = T(0.04045);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T gamma = T(2.4);

    // D65 white point
    const T Xn = T(0.95047);
    const T Yn = T(1.0);
    const T Zn = T(1.08883);

    // Lab constants
    const T delta = T(6.0 / 29.0);
    const T delta_cubed = delta * delta * delta;  // approximately 0.008856

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

    // Step 2: Linear RGB to XYZ
    const T r = linear[0];
    const T g = linear[1];
    const T b = linear[2];

    // Apply RGB to XYZ matrix (sRGB D65)
    // Matrix from IEC 61966-2-1:1999
    const T X = T(0.4124564) * r + T(0.3575761) * g + T(0.1804375) * b;
    const T Y = T(0.2126729) * r + T(0.7151522) * g + T(0.0721750) * b;
    const T Z = T(0.0193339) * r + T(0.1191920) * g + T(0.9503041) * b;

    // Step 3: XYZ to Lab using f(t) function
    // f(t) = t^(1/3)           if t > delta^3
    //      = t/(3*delta^2) + 4/29   otherwise
    const T tx = X / Xn;
    const T ty = Y / Yn;
    const T tz = Z / Zn;

    T fx, fy, fz;
    if (tx > delta_cubed) {
        fx = std::cbrt(tx);
    } else {
        fx = tx / (T(3) * delta * delta) + T(4.0 / 29.0);
    }
    if (ty > delta_cubed) {
        fy = std::cbrt(ty);
    } else {
        fy = ty / (T(3) * delta * delta) + T(4.0 / 29.0);
    }
    if (tz > delta_cubed) {
        fz = std::cbrt(tz);
    } else {
        fz = tz / (T(3) * delta * delta) + T(4.0 / 29.0);
    }

    lab[0] = T(116) * fy - T(16);       // L
    lab[1] = T(500) * (fx - fy);        // a
    lab[2] = T(200) * (fy - fz);        // b
}

}  // namespace torchscience::kernel::graphics::color
