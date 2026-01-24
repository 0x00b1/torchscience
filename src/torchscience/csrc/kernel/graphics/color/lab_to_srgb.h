#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single CIELAB pixel to sRGB.
 *
 * Implements CIELAB to sRGB conversion following these steps:
 * 1. Convert Lab to XYZ using the inverse f(t) function
 * 2. Apply the XYZ to RGB matrix for D65 illuminant
 * 3. Apply gamma encoding (companding)
 *
 * @param lab Input array [L, a, b] where L in [0,100], a,b typically in [-128,127]
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void lab_to_srgb_scalar(const T* lab, T* rgb) {
    // D65 white point
    const T Xn = T(0.95047);
    const T Yn = T(1.0);
    const T Zn = T(1.08883);

    // Lab constants
    const T delta = T(6.0 / 29.0);

    // sRGB encoding constants
    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    // Step 1: Lab to f values
    const T L = lab[0];
    const T a = lab[1];
    const T b = lab[2];

    const T fy = (L + T(16)) / T(116);
    const T fx = a / T(500) + fy;
    const T fz = fy - b / T(200);

    // Step 2: Inverse f function to get t values
    // f^-1(t) = t^3           if t > delta
    //         = 3*delta^2*(t - 4/29)  otherwise
    T tx, ty, tz;
    if (fx > delta) {
        tx = fx * fx * fx;
    } else {
        tx = T(3) * delta * delta * (fx - T(4.0 / 29.0));
    }
    if (fy > delta) {
        ty = fy * fy * fy;
    } else {
        ty = T(3) * delta * delta * (fy - T(4.0 / 29.0));
    }
    if (fz > delta) {
        tz = fz * fz * fz;
    } else {
        tz = T(3) * delta * delta * (fz - T(4.0 / 29.0));
    }

    // Step 3: Recover XYZ from normalized values
    const T X = Xn * tx;
    const T Y = Yn * ty;
    const T Z = Zn * tz;

    // Step 4: XYZ to linear RGB (inverse of RGB-to-XYZ matrix)
    const T r_linear = T( 3.2404542) * X + T(-1.5371385) * Y + T(-0.4985314) * Z;
    const T g_linear = T(-0.9692660) * X + T( 1.8760108) * Y + T( 0.0415560) * Z;
    const T b_linear = T( 0.0556434) * X + T(-0.2040259) * Y + T( 1.0572252) * Z;

    // Step 5: Linear RGB to sRGB (gamma encoding)
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
