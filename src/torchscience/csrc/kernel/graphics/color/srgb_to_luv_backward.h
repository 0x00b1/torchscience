#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to CIELUV conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. LUV output.
 * Uses chain rule through the composition: sRGB -> linear RGB -> XYZ -> LUV
 *
 * @param grad_luv Gradient w.r.t. LUV output [dL, du, dv]
 * @param rgb Original sRGB input [R, G, B]
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_luv_backward_scalar(const T* grad_luv, const T* rgb, T* grad_rgb) {
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
    const T delta_cubed = delta * delta * delta;

    // D65 reference chromaticity
    const T denom_n = Xn + T(15) * Yn + T(3) * Zn;
    const T u_prime_n = T(4) * Xn / denom_n;
    const T v_prime_n = T(9) * Yn / denom_n;

    // Forward pass to get intermediate values
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

    // Compute XYZ
    const T X = T(0.4124564) * r + T(0.3575761) * g + T(0.1804375) * b;
    const T Y = T(0.2126729) * r + T(0.7151522) * g + T(0.0721750) * b;
    const T Z = T(0.0193339) * r + T(0.1191920) * g + T(0.9503041) * b;

    const T yr = Y / Yn;

    // Compute L
    T L;
    if (yr > delta_cubed) {
        L = T(116) * std::cbrt(yr) - T(16);
    } else {
        L = T(903.3) * yr;
    }

    // Compute u' and v'
    const T denom = X + T(15) * Y + T(3) * Z;
    T u_prime = T(0);
    T v_prime = T(0);
    if (denom > T(0)) {
        u_prime = T(4) * X / denom;
        v_prime = T(9) * Y / denom;
    }

    // Gradients w.r.t. LUV
    const T dL = grad_luv[0];
    const T du_star = grad_luv[1];
    const T dv_star = grad_luv[2];

    // Derivatives of L w.r.t. yr
    // L = 116 * yr^(1/3) - 16  if yr > delta^3
    // L = 903.3 * yr           otherwise
    T dL_dyr;
    if (yr > delta_cubed) {
        // dL/dyr = 116 / (3 * yr^(2/3))
        dL_dyr = T(116) / (T(3) * std::cbrt(yr * yr));
    } else {
        dL_dyr = T(903.3);
    }

    // u* = 13 * L * (u' - u'n)
    // v* = 13 * L * (v' - v'n)
    //
    // Derivatives:
    // du*/dL = 13 * (u' - u'n)
    // du*/du' = 13 * L
    // dv*/dL = 13 * (v' - v'n)
    // dv*/dv' = 13 * L

    T grad_L_from_u_star = T(0);
    T grad_L_from_v_star = T(0);
    T grad_u_prime = T(0);
    T grad_v_prime = T(0);

    if (denom > T(0)) {
        grad_L_from_u_star = du_star * T(13) * (u_prime - u_prime_n);
        grad_L_from_v_star = dv_star * T(13) * (v_prime - v_prime_n);
        grad_u_prime = du_star * T(13) * L;
        grad_v_prime = dv_star * T(13) * L;
    }

    // Total gradient w.r.t. L
    const T grad_L_total = dL + grad_L_from_u_star + grad_L_from_v_star;

    // Gradient w.r.t. yr from L
    const T grad_yr = grad_L_total * dL_dyr;

    // Gradient w.r.t. Y from yr
    // yr = Y / Yn => dyr/dY = 1/Yn
    const T grad_Y_from_L = grad_yr / Yn;

    // Derivatives of u' and v' w.r.t. X, Y, Z
    // u' = 4X / (X + 15Y + 3Z)
    // v' = 9Y / (X + 15Y + 3Z)
    //
    // Let D = X + 15Y + 3Z
    // du'/dX = 4/D - 4X/D^2 = 4(D - X)/D^2 = 4(15Y + 3Z)/D^2
    // du'/dY = -4X * 15 / D^2 = -60X/D^2
    // du'/dZ = -4X * 3 / D^2 = -12X/D^2
    //
    // dv'/dX = -9Y / D^2
    // dv'/dY = 9/D - 9Y * 15 / D^2 = 9(D - 15Y)/D^2 = 9(X + 3Z)/D^2
    // dv'/dZ = -9Y * 3 / D^2 = -27Y/D^2

    T dX = T(0);
    T dY = T(0);
    T dZ = T(0);

    if (denom > T(0)) {
        const T inv_denom = T(1) / denom;
        const T inv_denom2 = inv_denom * inv_denom;

        // From u'
        const T du_prime_dX = T(4) * (T(15) * Y + T(3) * Z) * inv_denom2;
        const T du_prime_dY = -T(60) * X * inv_denom2;
        const T du_prime_dZ = -T(12) * X * inv_denom2;

        // From v'
        const T dv_prime_dX = -T(9) * Y * inv_denom2;
        const T dv_prime_dY = T(9) * (X + T(3) * Z) * inv_denom2;
        const T dv_prime_dZ = -T(27) * Y * inv_denom2;

        dX = grad_u_prime * du_prime_dX + grad_v_prime * dv_prime_dX;
        dY = grad_u_prime * du_prime_dY + grad_v_prime * dv_prime_dY;
        dZ = grad_u_prime * du_prime_dZ + grad_v_prime * dv_prime_dZ;
    }

    // Add gradient from L
    dY += grad_Y_from_L;

    // XYZ -> linear RGB (transpose of RGB-to-XYZ matrix)
    const T dr_linear = T(0.4124564) * dX + T(0.2126729) * dY + T(0.0193339) * dZ;
    const T dg_linear = T(0.3575761) * dX + T(0.7151522) * dY + T(0.1191920) * dZ;
    const T db_linear = T(0.1804375) * dX + T(0.0721750) * dY + T(0.9503041) * dZ;

    // linear RGB -> sRGB (derivative of linearization)
    T grad_linear[3] = {dr_linear, dg_linear, db_linear};
    for (int i = 0; i < 3; ++i) {
        const T v = rgb[i];
        T deriv;
        if (v <= threshold_srgb) {
            deriv = T(1) / linear_slope;
        } else {
            // d/dx [((x + offset) / scale)^gamma] = gamma/scale * ((x + offset) / scale)^(gamma-1)
            deriv = gamma / scale * std::pow((v + offset) / scale, gamma - T(1));
        }
        grad_rgb[i] = grad_linear[i] * deriv;
    }
}

}  // namespace torchscience::kernel::graphics::color
