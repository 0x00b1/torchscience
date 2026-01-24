#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to CIELAB conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. Lab output.
 * Uses chain rule through the composition: sRGB -> linear RGB -> XYZ -> Lab
 *
 * @param grad_lab Gradient w.r.t. Lab output [dL, da, db]
 * @param rgb Original sRGB input [R, G, B]
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_lab_backward_scalar(const T* grad_lab, const T* rgb, T* grad_rgb) {
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
    const T delta_cubed = delta * delta * delta;

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

    const T tx = X / Xn;
    const T ty = Y / Yn;
    const T tz = Z / Zn;

    // Derivative of f(t)
    // f'(t) = 1/(3 * t^(2/3))  if t > delta^3
    //       = 1/(3*delta^2)     otherwise
    T dfx_raw, dfy_raw, dfz_raw;
    if (tx > delta_cubed) {
        dfx_raw = T(1) / (T(3) * std::cbrt(tx * tx));
    } else {
        dfx_raw = T(1) / (T(3) * delta * delta);
    }
    if (ty > delta_cubed) {
        dfy_raw = T(1) / (T(3) * std::cbrt(ty * ty));
    } else {
        dfy_raw = T(1) / (T(3) * delta * delta);
    }
    if (tz > delta_cubed) {
        dfz_raw = T(1) / (T(3) * std::cbrt(tz * tz));
    } else {
        dfz_raw = T(1) / (T(3) * delta * delta);
    }

    // Derivatives of f(t/Tn) w.r.t. t = df/dt * (1/Tn)
    const T dfx = dfx_raw / Xn;
    const T dfy = dfy_raw / Yn;
    const T dfz = dfz_raw / Zn;

    // Gradients w.r.t. Lab
    const T dL = grad_lab[0];
    const T da = grad_lab[1];
    const T db = grad_lab[2];

    // Chain rule: Lab -> f values
    // L = 116*fy - 16  =>  grad_fy from L = 116 * dL
    // a = 500*(fx-fy)  =>  grad_fx from a = 500*da, grad_fy from a = -500*da
    // b = 200*(fy-fz)  =>  grad_fy from b = 200*db, grad_fz from b = -200*db
    const T grad_fx = T(500) * da;
    const T grad_fy = T(116) * dL - T(500) * da + T(200) * db;
    const T grad_fz = -T(200) * db;

    // XYZ gradients (chain through f and division by white point)
    const T dX = grad_fx * dfx;
    const T dY = grad_fy * dfy;
    const T dZ = grad_fz * dfz;

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
