#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for XYZ to sRGB conversion.
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param xyz Original XYZ input [X, Y, Z]
 * @param grad_xyz Output gradient w.r.t. XYZ [dX, dY, dZ]
 */
template <typename T>
void xyz_to_srgb_backward_scalar(const T* grad_rgb, const T* xyz, T* grad_xyz) {
    const T X = xyz[0];
    const T Y = xyz[1];
    const T Z = xyz[2];

    // Compute linear RGB (needed for derivative)
    const T r_linear = T( 3.2404542) * X + T(-1.5371385) * Y + T(-0.4985314) * Z;
    const T g_linear = T(-0.9692660) * X + T( 1.8760108) * Y + T( 0.0415560) * Z;
    const T b_linear = T( 0.0556434) * X + T(-0.2040259) * Y + T( 1.0572252) * Z;

    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T scale = T(1.055);
    const T inv_gamma = T(1.0 / 2.4);

    // Compute derivative of gamma encoding for each channel
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

    // grad_linear = grad_rgb * gamma_deriv (element-wise)
    const T dr_linear = grad_rgb[0] * gamma_deriv[0];
    const T dg_linear = grad_rgb[1] * gamma_deriv[1];
    const T db_linear = grad_rgb[2] * gamma_deriv[2];

    // grad_xyz = M^T * grad_linear (transpose of XYZ-to-RGB matrix)
    grad_xyz[0] = T( 3.2404542) * dr_linear + T(-0.9692660) * dg_linear + T( 0.0556434) * db_linear;
    grad_xyz[1] = T(-1.5371385) * dr_linear + T( 1.8760108) * dg_linear + T(-0.2040259) * db_linear;
    grad_xyz[2] = T(-0.4985314) * dr_linear + T( 0.0415560) * dg_linear + T( 1.0572252) * db_linear;
}

}  // namespace torchscience::kernel::graphics::color
