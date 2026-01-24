#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to Oklab conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. Oklab output.
 * Uses chain rule through the composition: sRGB -> linear RGB -> LMS -> L'M'S' -> Lab
 *
 * @param grad_lab Gradient w.r.t. Oklab output [dL, da, db]
 * @param rgb Original sRGB input [R, G, B]
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_oklab_backward_scalar(const T* grad_lab, const T* rgb, T* grad_rgb) {
    // sRGB linearization constants
    const T threshold_srgb = T(0.04045);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T gamma = T(2.4);

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

    // Compute LMS
    const T l = T(0.4122214708) * r + T(0.5363325363) * g + T(0.0514459929) * b;
    const T m = T(0.2119034982) * r + T(0.6806995451) * g + T(0.1073969566) * b;
    const T s = T(0.0883024619) * r + T(0.2817188376) * g + T(0.6299787005) * b;

    // Gradients w.r.t. Oklab
    const T dL = grad_lab[0];
    const T da = grad_lab[1];
    const T db = grad_lab[2];

    // Chain rule: Lab -> L'M'S' (transpose of M2)
    // M2 = [[0.2104542553, 0.7936177850, -0.0040720468],
    //       [1.9779984951, -2.4285922050, 0.4505937099],
    //       [0.0259040371, 0.7827717662, -0.8086757660]]
    const T grad_l_prime = T(0.2104542553) * dL + T(1.9779984951) * da + T(0.0259040371) * db;
    const T grad_m_prime = T(0.7936177850) * dL - T(2.4285922050) * da + T(0.7827717662) * db;
    const T grad_s_prime = -T(0.0040720468) * dL + T(0.4505937099) * da - T(0.8086757660) * db;

    // Chain rule: L'M'S' -> LMS
    // Derivative of cbrt(x) = 1/(3 * x^(2/3)) = 1/(3 * cbrt(x)^2)
    // For x < 0: cbrt(x) = -cbrt(-x), d/dx cbrt(x) = 1/(3 * cbrt(x^2))
    // std::cbrt handles negative values, derivative is 1/(3 * cbrt(x)^2)
    T dl_dlms, dm_dlms, ds_dlms;

    // Handle the case when l, m, s are close to zero
    const T epsilon = T(1e-12);

    if (std::abs(l) > epsilon) {
        const T cbrt_l = std::cbrt(l);
        dl_dlms = T(1) / (T(3) * cbrt_l * cbrt_l);
    } else {
        // For very small values, use a large but finite derivative
        dl_dlms = T(1) / (T(3) * std::cbrt(epsilon * epsilon));
    }

    if (std::abs(m) > epsilon) {
        const T cbrt_m = std::cbrt(m);
        dm_dlms = T(1) / (T(3) * cbrt_m * cbrt_m);
    } else {
        dm_dlms = T(1) / (T(3) * std::cbrt(epsilon * epsilon));
    }

    if (std::abs(s) > epsilon) {
        const T cbrt_s = std::cbrt(s);
        ds_dlms = T(1) / (T(3) * cbrt_s * cbrt_s);
    } else {
        ds_dlms = T(1) / (T(3) * std::cbrt(epsilon * epsilon));
    }

    const T grad_l = grad_l_prime * dl_dlms;
    const T grad_m = grad_m_prime * dm_dlms;
    const T grad_s = grad_s_prime * ds_dlms;

    // Chain rule: LMS -> linear RGB (transpose of M1)
    // M1 = [[0.4122214708, 0.5363325363, 0.0514459929],
    //       [0.2119034982, 0.6806995451, 0.1073969566],
    //       [0.0883024619, 0.2817188376, 0.6299787005]]
    const T dr_linear = T(0.4122214708) * grad_l + T(0.2119034982) * grad_m + T(0.0883024619) * grad_s;
    const T dg_linear = T(0.5363325363) * grad_l + T(0.6806995451) * grad_m + T(0.2817188376) * grad_s;
    const T db_linear = T(0.0514459929) * grad_l + T(0.1073969566) * grad_m + T(0.6299787005) * grad_s;

    // Chain rule: linear RGB -> sRGB (derivative of linearization)
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
