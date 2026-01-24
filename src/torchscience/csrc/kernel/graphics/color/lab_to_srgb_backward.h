#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for CIELAB to sRGB conversion.
 *
 * Computes gradients w.r.t. Lab input given gradients w.r.t. sRGB output.
 * Uses chain rule through the composition: Lab -> XYZ -> linear RGB -> sRGB
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param lab Original Lab input [L, a, b]
 * @param grad_lab Output gradient w.r.t. Lab [dL, da, db]
 */
template <typename T>
void lab_to_srgb_backward_scalar(const T* grad_rgb, const T* lab, T* grad_lab) {
    // D65 white point
    const T Xn = T(0.95047);
    const T Yn = T(1.0);
    const T Zn = T(1.08883);

    // Lab constants
    const T delta = T(6.0 / 29.0);

    // sRGB encoding constants
    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    // Forward pass to get intermediate values
    const T L = lab[0];
    const T a = lab[1];
    const T b = lab[2];

    const T fy = (L + T(16)) / T(116);
    const T fx = a / T(500) + fy;
    const T fz = fy - b / T(200);

    // Compute t values and their derivatives
    T tx, ty, tz;
    T dtx_dfx, dty_dfy, dtz_dfz;

    if (fx > delta) {
        tx = fx * fx * fx;
        dtx_dfx = T(3) * fx * fx;
    } else {
        tx = T(3) * delta * delta * (fx - T(4.0 / 29.0));
        dtx_dfx = T(3) * delta * delta;
    }
    if (fy > delta) {
        ty = fy * fy * fy;
        dty_dfy = T(3) * fy * fy;
    } else {
        ty = T(3) * delta * delta * (fy - T(4.0 / 29.0));
        dty_dfy = T(3) * delta * delta;
    }
    if (fz > delta) {
        tz = fz * fz * fz;
        dtz_dfz = T(3) * fz * fz;
    } else {
        tz = T(3) * delta * delta * (fz - T(4.0 / 29.0));
        dtz_dfz = T(3) * delta * delta;
    }

    // Compute XYZ
    const T X = Xn * tx;
    const T Y = Yn * ty;
    const T Z = Zn * tz;

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

    // Backprop: XYZ -> t values (X = Xn * tx, etc.)
    const T dtx = dX * Xn;
    const T dty = dY * Yn;
    const T dtz = dZ * Zn;

    // Backprop: t -> f values
    const T dfx = dtx * dtx_dfx;
    const T dfy = dty * dty_dfy;
    const T dfz = dtz * dtz_dfz;

    // Backprop: f values -> Lab
    // fy = (L + 16) / 116  =>  dfy/dL = 1/116
    // fx = a / 500 + fy    =>  dfx/da = 1/500, dfx/dL = 1/116
    // fz = fy - b / 200    =>  dfz/db = -1/200, dfz/dL = 1/116
    //
    // grad_L = dfx * (1/116) + dfy * (1/116) + dfz * (1/116)
    // grad_a = dfx * (1/500)
    // grad_b = dfz * (-1/200)

    grad_lab[0] = (dfx + dfy + dfz) / T(116);
    grad_lab[1] = dfx / T(500);
    grad_lab[2] = -dfz / T(200);
}

}  // namespace torchscience::kernel::graphics::color
