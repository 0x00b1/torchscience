#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for CIELUV to sRGB conversion.
 *
 * Computes gradients w.r.t. LUV input given gradients w.r.t. sRGB output.
 * Uses chain rule through the composition: LUV -> XYZ -> linear RGB -> sRGB
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param luv Original LUV input [L, u, v]
 * @param grad_luv Output gradient w.r.t. LUV [dL, du, dv]
 */
template <typename T>
void luv_to_srgb_backward_scalar(const T* grad_rgb, const T* luv, T* grad_luv) {
    // D65 white point
    const T Xn = T(0.95047);
    const T Yn = T(1.0);
    const T Zn = T(1.08883);

    // D65 reference chromaticity
    const T denom_n = Xn + T(15) * Yn + T(3) * Zn;
    const T u_prime_n = T(4) * Xn / denom_n;
    const T v_prime_n = T(9) * Yn / denom_n;

    // sRGB encoding constants
    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    // Extract LUV values
    const T L = luv[0];
    const T u_star = luv[1];
    const T v_star = luv[2];

    // Forward pass to get intermediate values
    T u_prime, v_prime;
    if (L > T(0)) {
        u_prime = u_star / (T(13) * L) + u_prime_n;
        v_prime = v_star / (T(13) * L) + v_prime_n;
    } else {
        u_prime = u_prime_n;
        v_prime = v_prime_n;
    }

    // Compute Y
    T Y;
    T dY_dL;  // derivative of Y w.r.t. L
    if (L > T(8)) {
        const T t = (L + T(16)) / T(116);
        Y = Yn * t * t * t;
        // dY/dL = Yn * 3 * t^2 * (1/116)
        dY_dL = Yn * T(3) * t * t / T(116);
    } else {
        Y = Yn * L / T(903.3);
        dY_dL = Yn / T(903.3);
    }

    // Compute X and Z
    T X, Z;
    if (v_prime > T(0)) {
        X = Y * T(9) * u_prime / (T(4) * v_prime);
        Z = Y * (T(12) - T(3) * u_prime - T(20) * v_prime) / (T(4) * v_prime);
    } else {
        X = T(0);
        Z = T(0);
    }

    // Compute linear RGB
    const T r_linear = T( 3.2404542) * X + T(-1.5371385) * Y + T(-0.4985314) * Z;
    const T g_linear = T(-0.9692660) * X + T( 1.8760108) * Y + T( 0.0415560) * Z;
    const T b_linear = T( 0.0556434) * X + T(-0.2040259) * Y + T( 1.0572252) * Z;

    // Compute gamma encoding derivatives
    T linear[3] = {r_linear, g_linear, b_linear};
    T gamma_deriv[3];
    for (int i = 0; i < 3; ++i) {
        const T value = linear[i];
        if (value <= threshold) {
            gamma_deriv[i] = linear_slope;
        } else {
            // d/dx [scale * x^inv_gamma - offset] = scale * inv_gamma * x^(inv_gamma - 1)
            gamma_deriv[i] = scale * inv_gamma * std::pow(value, inv_gamma - T(1));
        }
    }

    // Backprop: sRGB -> linear RGB
    const T dr_linear = grad_rgb[0] * gamma_deriv[0];
    const T dg_linear = grad_rgb[1] * gamma_deriv[1];
    const T db_linear = grad_rgb[2] * gamma_deriv[2];

    // Backprop: linear RGB -> XYZ (transpose of XYZ-to-RGB matrix)
    const T dX = T( 3.2404542) * dr_linear + T(-0.9692660) * dg_linear + T( 0.0556434) * db_linear;
    const T dY = T(-1.5371385) * dr_linear + T( 1.8760108) * dg_linear + T(-0.2040259) * db_linear;
    const T dZ = T(-0.4985314) * dr_linear + T( 0.0415560) * dg_linear + T( 1.0572252) * db_linear;

    // Derivatives of X, Z w.r.t. Y, u', v'
    // X = Y * 9 * u' / (4 * v') = (9/4) * Y * u' / v'
    // Z = Y * (12 - 3*u' - 20*v') / (4 * v')
    //
    // dX/dY = 9 * u' / (4 * v')
    // dX/du' = 9 * Y / (4 * v')
    // dX/dv' = -9 * Y * u' / (4 * v'^2)
    //
    // dZ/dY = (12 - 3*u' - 20*v') / (4 * v')
    // dZ/du' = -3 * Y / (4 * v')
    // dZ/dv' = Y * (-20 * 4 * v' - (12 - 3*u' - 20*v') * 4) / (16 * v'^2)
    //        = Y * (-80*v' - 48 + 12*u' + 80*v') / (16 * v'^2)
    //        = Y * (12*u' - 48) / (16 * v'^2)
    //        = Y * (u' - 4) * 3 / (4 * v'^2)

    T grad_Y_from_XZ = T(0);
    T grad_u_prime = T(0);
    T grad_v_prime = T(0);

    if (v_prime > T(0)) {
        const T inv_v_prime = T(1) / v_prime;
        const T inv_v_prime2 = inv_v_prime * inv_v_prime;

        // From X
        const T dX_dY = T(9) * u_prime / (T(4) * v_prime);
        const T dX_du_prime = T(9) * Y / (T(4) * v_prime);
        const T dX_dv_prime = -T(9) * Y * u_prime / (T(4) * v_prime * v_prime);

        // From Z
        const T dZ_dY = (T(12) - T(3) * u_prime - T(20) * v_prime) / (T(4) * v_prime);
        const T dZ_du_prime = -T(3) * Y / (T(4) * v_prime);
        const T dZ_dv_prime = T(3) * Y * (u_prime - T(4)) / (T(4) * v_prime * v_prime);

        grad_Y_from_XZ = dX * dX_dY + dZ * dZ_dY;
        grad_u_prime = dX * dX_du_prime + dZ * dZ_du_prime;
        grad_v_prime = dX * dX_dv_prime + dZ * dZ_dv_prime;
    }

    // Total gradient w.r.t. Y
    const T grad_Y_total = dY + grad_Y_from_XZ;

    // Derivatives of u', v' w.r.t. L, u*, v*
    // u' = u* / (13 * L) + u'n
    // v' = v* / (13 * L) + v'n
    //
    // du'/dL = -u* / (13 * L^2)
    // du'/du* = 1 / (13 * L)
    // dv'/dL = -v* / (13 * L^2)
    // dv'/dv* = 1 / (13 * L)

    T grad_L = T(0);
    T grad_u_star = T(0);
    T grad_v_star = T(0);

    if (L > T(0)) {
        const T inv_L = T(1) / L;
        const T inv_L2 = inv_L * inv_L;

        const T du_prime_dL = -u_star / (T(13) * L * L);
        const T du_prime_du_star = T(1) / (T(13) * L);
        const T dv_prime_dL = -v_star / (T(13) * L * L);
        const T dv_prime_dv_star = T(1) / (T(13) * L);

        grad_L = grad_u_prime * du_prime_dL + grad_v_prime * dv_prime_dL;
        grad_u_star = grad_u_prime * du_prime_du_star;
        grad_v_star = grad_v_prime * dv_prime_dv_star;
    }

    // Add gradient from Y to L
    grad_L += grad_Y_total * dY_dL;

    grad_luv[0] = grad_L;
    grad_luv[1] = grad_u_star;
    grad_luv[2] = grad_v_star;
}

}  // namespace torchscience::kernel::graphics::color
