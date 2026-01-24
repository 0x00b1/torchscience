#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to XYZ conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. XYZ output.
 * Uses chain rule: grad_rgb = (M^T * grad_xyz) * linearize'(rgb)
 * where M is the RGB-to-XYZ matrix.
 *
 * @param grad_xyz Gradient w.r.t. XYZ output [dX, dY, dZ]
 * @param rgb Original sRGB input [R, G, B]
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_xyz_backward_scalar(const T* grad_xyz, const T* rgb, T* grad_rgb) {
    const T threshold = T(0.04045);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);

    // Transpose of RGB to XYZ matrix times grad_xyz
    const T dX = grad_xyz[0];
    const T dY = grad_xyz[1];
    const T dZ = grad_xyz[2];

    // grad w.r.t. linear RGB (matrix transpose)
    const T dr_linear = T(0.4124564) * dX + T(0.2126729) * dY + T(0.0193339) * dZ;
    const T dg_linear = T(0.3575761) * dX + T(0.7151522) * dY + T(0.1191920) * dZ;
    const T db_linear = T(0.1804375) * dX + T(0.0721750) * dY + T(0.9503041) * dZ;

    const T grad_linear[3] = {dr_linear, dg_linear, db_linear};

    // Chain rule with linearization derivative
    for (int i = 0; i < 3; ++i) {
        const T value = rgb[i];
        T deriv;
        if (value <= threshold) {
            deriv = T(1) / linear_slope;
        } else {
            // d/dx [((x + offset) / scale)^gamma] = gamma/scale * ((x + offset) / scale)^(gamma-1)
            deriv = T(2.4) / scale * std::pow((value + offset) / scale, T(1.4));
        }
        grad_rgb[i] = grad_linear[i] * deriv;
    }
}

}  // namespace torchscience::kernel::graphics::color
