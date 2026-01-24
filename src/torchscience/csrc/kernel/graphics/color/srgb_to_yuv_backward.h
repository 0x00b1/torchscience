#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to YUV conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. YUV output.
 * Since this is a linear transformation, the gradient is just the transpose
 * of the forward matrix.
 *
 * @param grad_yuv Gradient w.r.t. YUV output [dY, dU, dV]
 * @param rgb Original sRGB input [R, G, B] (not used for linear transform)
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_yuv_backward_scalar(const T* grad_yuv, const T* /*rgb*/, T* grad_rgb) {
    const T dY = grad_yuv[0];
    const T dU = grad_yuv[1];
    const T dV = grad_yuv[2];

    // Transpose of forward matrix:
    // Forward:  Y =  0.299*R          + 0.587*G          + 0.114*B
    //           U = -0.1471376975*R   - 0.2888623025*G   + 0.436*B
    //           V =  0.615*R          - 0.5149857347*G   - 0.1000142653*B
    //
    // Transpose: dR = 0.299*dY - 0.1471376975*dU + 0.615*dV
    //            dG = 0.587*dY - 0.2888623025*dU - 0.5149857347*dV
    //            dB = 0.114*dY + 0.436*dU - 0.1000142653*dV
    grad_rgb[0] = T(0.299) * dY + T(-0.1471376975) * dU + T(0.615) * dV;
    grad_rgb[1] = T(0.587) * dY + T(-0.2888623025) * dU + T(-0.5149857347) * dV;
    grad_rgb[2] = T(0.114) * dY + T(0.436) * dU + T(-0.1000142653) * dV;
}

}  // namespace torchscience::kernel::graphics::color
