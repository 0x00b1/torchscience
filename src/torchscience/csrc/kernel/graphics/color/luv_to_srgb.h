#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single CIELUV pixel to sRGB.
 *
 * Implements CIELUV to sRGB conversion following these steps:
 * 1. Convert LUV to XYZ using the inverse u', v' chromaticity coordinates
 * 2. Apply the XYZ to RGB matrix for D65 illuminant
 * 3. Apply gamma encoding (companding)
 *
 * @param luv Input array [L, u, v] where L in [0,100], u,v typically in [-100,100]
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void luv_to_srgb_scalar(const T* luv, T* rgb) {
    // D65 white point
    const T Xn = T(0.95047);
    const T Yn = T(1.0);
    const T Zn = T(1.08883);

    // LUV constants
    const T delta = T(6.0 / 29.0);

    // D65 reference chromaticity
    const T denom_n = Xn + T(15) * Yn + T(3) * Zn;
    const T u_prime_n = T(4) * Xn / denom_n;
    const T v_prime_n = T(9) * Yn / denom_n;

    // sRGB encoding constants
    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    // Step 1: Extract LUV values
    const T L = luv[0];
    const T u_star = luv[1];
    const T v_star = luv[2];

    // Step 2: Compute u' and v' from u* and v*
    // u* = 13 * L * (u' - u'n)
    // v* = 13 * L * (v' - v'n)
    // => u' = u* / (13 * L) + u'n
    // => v' = v* / (13 * L) + v'n
    T u_prime, v_prime;
    if (L > T(0)) {
        u_prime = u_star / (T(13) * L) + u_prime_n;
        v_prime = v_star / (T(13) * L) + v_prime_n;
    } else {
        // At L=0 (black), u' and v' are undefined
        // Use reference white point chromaticity by convention
        u_prime = u_prime_n;
        v_prime = v_prime_n;
    }

    // Step 3: Compute Y from L
    // L* = 116 * (Y/Yn)^(1/3) - 16  if Y/Yn > delta^3
    // L* = (29/3)^3 * Y/Yn          otherwise
    // => Y = Yn * ((L* + 16) / 116)^3  if L* > 8
    // => Y = Yn * L* / 903.3          otherwise
    T Y;
    if (L > T(8)) {
        const T t = (L + T(16)) / T(116);
        Y = Yn * t * t * t;
    } else {
        Y = Yn * L / T(903.3);
    }

    // Step 4: Compute X and Z from Y, u', v'
    // u' = 4X / (X + 15Y + 3Z)
    // v' = 9Y / (X + 15Y + 3Z)
    //
    // From v': X + 15Y + 3Z = 9Y / v'
    // From u': X = u' * (X + 15Y + 3Z) / 4 = u' * 9Y / (4 * v')
    //
    // X = 9 * Y * u' / (4 * v')
    // Z = (X + 15Y + 3Z - X - 15Y) / 3 = ((9Y/v') - X - 15Y) / 3
    //   = (9Y/v' - 9Yu'/(4v') - 15Y) / 3
    //   = Y * (9/v' - 9u'/(4v') - 15) / 3
    //   = Y * (36 - 9u' - 60v') / (12v')
    //   = Y * (12 - 3u' - 20v') / (4v')
    T X, Z;
    if (v_prime > T(0)) {
        X = Y * T(9) * u_prime / (T(4) * v_prime);
        Z = Y * (T(12) - T(3) * u_prime - T(20) * v_prime) / (T(4) * v_prime);
    } else {
        // v' = 0 means we're at a singularity
        // This can occur for very dark colors
        X = T(0);
        Z = T(0);
    }

    // Step 5: XYZ to linear RGB (inverse of RGB-to-XYZ matrix)
    const T r_linear = T( 3.2404542) * X + T(-1.5371385) * Y + T(-0.4985314) * Z;
    const T g_linear = T(-0.9692660) * X + T( 1.8760108) * Y + T( 0.0415560) * Z;
    const T b_linear = T( 0.0556434) * X + T(-0.2040259) * Y + T( 1.0572252) * Z;

    // Step 6: Linear RGB to sRGB (gamma encoding)
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
