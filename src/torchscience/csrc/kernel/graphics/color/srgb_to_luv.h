#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to CIELUV.
 *
 * Implements sRGB to CIELUV conversion following these steps:
 * 1. Linearize sRGB using the standard transfer function (IEC 61966-2-1:1999)
 * 2. Apply the RGB to XYZ matrix for D65 illuminant
 * 3. Convert XYZ to LUV using the u', v' chromaticity coordinates
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param luv Output array [L, u, v] where L in [0,100], u,v typically in [-100,100]
 */
template <typename T>
void srgb_to_luv_scalar(const T* rgb, T* luv) {
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

    // LUV constants
    const T delta = T(6.0 / 29.0);
    const T delta_cubed = delta * delta * delta;  // approximately 0.008856

    // D65 reference chromaticity
    // u'n = 4 * Xn / (Xn + 15 * Yn + 3 * Zn)
    // v'n = 9 * Yn / (Xn + 15 * Yn + 3 * Zn)
    const T denom_n = Xn + T(15) * Yn + T(3) * Zn;
    const T u_prime_n = T(4) * Xn / denom_n;  // approximately 0.19784
    const T v_prime_n = T(9) * Yn / denom_n;  // approximately 0.46834

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

    // Step 3: XYZ to LUV
    // Compute L*
    const T yr = Y / Yn;
    T L;
    if (yr > delta_cubed) {
        L = T(116) * std::cbrt(yr) - T(16);
    } else {
        // L* = (29/3)^3 * Y/Yn = 903.3 * Y/Yn
        L = T(903.3) * yr;
    }

    // Compute u' and v' chromaticity
    const T denom = X + T(15) * Y + T(3) * Z;

    T u_star, v_star;
    if (denom > T(0)) {
        const T u_prime = T(4) * X / denom;
        const T v_prime = T(9) * Y / denom;

        u_star = T(13) * L * (u_prime - u_prime_n);
        v_star = T(13) * L * (v_prime - v_prime_n);
    } else {
        // At black point (X=Y=Z=0), u* and v* are undefined
        // By convention, set them to 0
        u_star = T(0);
        v_star = T(0);
    }

    luv[0] = L;
    luv[1] = u_star;
    luv[2] = v_star;
}

}  // namespace torchscience::kernel::graphics::color
