#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to YPbPr conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. YPbPr output.
 * Uses chain rule: d_loss/d_srgb = d_loss/d_ypbpr * d_ypbpr/d_linear * d_linear/d_srgb
 *
 * The forward path is: sRGB -> linear RGB -> YPbPr
 *
 * @param grad_ypbpr Gradient w.r.t. YPbPr output [dY, dPb, dPr]
 * @param rgb Original sRGB input [R, G, B]
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_ypbpr_backward_scalar(const T* grad_ypbpr, const T* rgb, T* grad_rgb) {
    const T dY = grad_ypbpr[0];
    const T dPb = grad_ypbpr[1];
    const T dPr = grad_ypbpr[2];

    // Gradient of YPbPr w.r.t. linear RGB (transpose of forward matrix)
    // Forward:  Y  =  0.299*R_lin     + 0.587*G_lin     + 0.114*B_lin
    //           Pb = -0.168736*R_lin  - 0.331264*G_lin  + 0.5*B_lin
    //           Pr =  0.5*R_lin       - 0.418688*G_lin  - 0.081312*B_lin
    //
    // Transpose: dR_lin = 0.299*dY - 0.168736*dPb + 0.5*dPr
    //            dG_lin = 0.587*dY - 0.331264*dPb - 0.418688*dPr
    //            dB_lin = 0.114*dY + 0.5*dPb - 0.081312*dPr
    const T dR_lin = T(0.299) * dY + T(-0.168736) * dPb + T(0.5) * dPr;
    const T dG_lin = T(0.587) * dY + T(-0.331264) * dPb + T(-0.418688) * dPr;
    const T dB_lin = T(0.114) * dY + T(0.5) * dPb + T(-0.081312) * dPr;

    // Gradient of sRGB to linear transformation
    // if srgb <= 0.04045: d_linear/d_srgb = 1 / 12.92
    // else: d_linear/d_srgb = (2.4 / 1.055) * ((srgb + 0.055) / 1.055)^1.4
    const T threshold = T(0.04045);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T gamma = T(2.4);
    const T gamma_minus_one = T(1.4);

    T local_grads[3];
    for (int i = 0; i < 3; ++i) {
        const T value = rgb[i];
        if (value <= threshold) {
            local_grads[i] = T(1) / linear_slope;
        } else {
            local_grads[i] = (gamma / scale) * std::pow((value + offset) / scale, gamma_minus_one);
        }
    }

    // Chain rule: d_loss/d_srgb = d_loss/d_linear * d_linear/d_srgb
    grad_rgb[0] = dR_lin * local_grads[0];
    grad_rgb[1] = dG_lin * local_grads[1];
    grad_rgb[2] = dB_lin * local_grads[2];
}

}  // namespace torchscience::kernel::graphics::color
