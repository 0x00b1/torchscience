#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for YCbCr to sRGB conversion.
 *
 * Computes gradients w.r.t. YCbCr input given gradients w.r.t. sRGB output.
 * Since this is a linear transformation, the gradient is just the transpose
 * of the forward matrix.
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param ycbcr Original YCbCr input [Y, Cb, Cr] (not used for linear transform)
 * @param grad_ycbcr Output gradient w.r.t. YCbCr [dY, dCb, dCr]
 */
template <typename T>
void ycbcr_to_srgb_backward_scalar(const T* grad_rgb, const T* /*ycbcr*/, T* grad_ycbcr) {
    const T dR = grad_rgb[0];
    const T dG = grad_rgb[1];
    const T dB = grad_rgb[2];

    // Transpose of forward matrix:
    // Forward:  R = Y + 0*Cb + 1.402*Cr
    //           G = Y - 0.344136*Cb - 0.714136*Cr
    //           B = Y + 1.772*Cb + 0*Cr
    //
    // Transpose: dY  = dR + dG + dB
    //            dCb = 0*dR - 0.344136*dG + 1.772*dB
    //            dCr = 1.402*dR - 0.714136*dG + 0*dB
    grad_ycbcr[0] = dR + dG + dB;
    grad_ycbcr[1] = T(-0.344136) * dG + T(1.772) * dB;
    grad_ycbcr[2] = T(1.402) * dR + T(-0.714136) * dG;
}

}  // namespace torchscience::kernel::graphics::color
