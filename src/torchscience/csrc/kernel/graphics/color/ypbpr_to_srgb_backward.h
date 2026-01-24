#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for YPbPr to sRGB conversion.
 *
 * Computes gradients w.r.t. YPbPr input given gradients w.r.t. sRGB output.
 * Uses chain rule: d_loss/d_ypbpr = d_loss/d_srgb * d_srgb/d_linear * d_linear/d_ypbpr
 *
 * The forward path is: YPbPr -> linear RGB -> sRGB
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param ypbpr Original YPbPr input [Y, Pb, Pr]
 * @param grad_ypbpr Output gradient w.r.t. YPbPr [dY, dPb, dPr]
 */
template <typename T>
void ypbpr_to_srgb_backward_scalar(const T* grad_rgb, const T* ypbpr, T* grad_ypbpr) {
    const T y = ypbpr[0];
    const T pb = ypbpr[1];
    const T pr = ypbpr[2];

    // Compute linear RGB values (needed for gamma derivative)
    const T r_lin = y + T(1.402) * pr;
    const T g_lin = y + T(-0.344136) * pb + T(-0.714136) * pr;
    const T b_lin = y + T(1.772) * pb;

    // Gradient of linear to sRGB transformation
    // if linear <= 0.0031308: d_srgb/d_linear = 12.92
    // else: d_srgb/d_linear = (1.055 / 2.4) * linear^(1/2.4 - 1)
    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T scale = T(1.055);
    const T gamma = T(2.4);
    const T inverse_gamma_minus_one = T(1.0 / 2.4 - 1.0);

    T gamma_grads[3];
    T linear_vals[3] = {r_lin, g_lin, b_lin};
    for (int i = 0; i < 3; ++i) {
        const T value = linear_vals[i];
        if (value <= threshold) {
            gamma_grads[i] = linear_slope;
        } else {
            gamma_grads[i] = (scale / gamma) * std::pow(value, inverse_gamma_minus_one);
        }
    }

    // Gradient w.r.t. linear RGB
    const T dR_lin = grad_rgb[0] * gamma_grads[0];
    const T dG_lin = grad_rgb[1] * gamma_grads[1];
    const T dB_lin = grad_rgb[2] * gamma_grads[2];

    // Transpose of YPbPr to linear RGB matrix:
    // Forward:  R_lin = Y + 0*Pb + 1.402*Pr
    //           G_lin = Y - 0.344136*Pb - 0.714136*Pr
    //           B_lin = Y + 1.772*Pb + 0*Pr
    //
    // Transpose: dY  = dR_lin + dG_lin + dB_lin
    //            dPb = 0*dR_lin - 0.344136*dG_lin + 1.772*dB_lin
    //            dPr = 1.402*dR_lin - 0.714136*dG_lin + 0*dB_lin
    grad_ypbpr[0] = dR_lin + dG_lin + dB_lin;
    grad_ypbpr[1] = T(-0.344136) * dG_lin + T(1.772) * dB_lin;
    grad_ypbpr[2] = T(1.402) * dR_lin + T(-0.714136) * dG_lin;
}

}  // namespace torchscience::kernel::graphics::color
