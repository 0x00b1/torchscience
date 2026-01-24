#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to YCbCr conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. YCbCr output.
 * Since this is a linear transformation, the gradient is just the transpose
 * of the forward matrix.
 *
 * @param grad_ycbcr Gradient w.r.t. YCbCr output [dY, dCb, dCr]
 * @param rgb Original sRGB input [R, G, B] (not used for linear transform)
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_ycbcr_backward_scalar(const T* grad_ycbcr, const T* /*rgb*/, T* grad_rgb) {
    const T dY = grad_ycbcr[0];
    const T dCb = grad_ycbcr[1];
    const T dCr = grad_ycbcr[2];

    // Transpose of forward matrix:
    // Forward:  Y  =  0.299*R     + 0.587*G     + 0.114*B
    //           Cb = -0.168736*R  - 0.331264*G  + 0.5*B
    //           Cr =  0.5*R       - 0.418688*G  - 0.081312*B
    //
    // Transpose: dR = 0.299*dY - 0.168736*dCb + 0.5*dCr
    //            dG = 0.587*dY - 0.331264*dCb - 0.418688*dCr
    //            dB = 0.114*dY + 0.5*dCb - 0.081312*dCr
    grad_rgb[0] = T(0.299) * dY + T(-0.168736) * dCb + T(0.5) * dCr;
    grad_rgb[1] = T(0.587) * dY + T(-0.331264) * dCb + T(-0.418688) * dCr;
    grad_rgb[2] = T(0.114) * dY + T(0.5) * dCb + T(-0.081312) * dCr;
}

}  // namespace torchscience::kernel::graphics::color
