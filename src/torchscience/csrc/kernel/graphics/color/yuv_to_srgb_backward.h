#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for YUV to sRGB conversion.
 *
 * Computes gradients w.r.t. YUV input given gradients w.r.t. sRGB output.
 * Since this is a linear transformation, the gradient is just the transpose
 * of the forward matrix.
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param yuv Original YUV input [Y, U, V] (not used for linear transform)
 * @param grad_yuv Output gradient w.r.t. YUV [dY, dU, dV]
 */
template <typename T>
void yuv_to_srgb_backward_scalar(const T* grad_rgb, const T* /*yuv*/, T* grad_yuv) {
    const T dR = grad_rgb[0];
    const T dG = grad_rgb[1];
    const T dB = grad_rgb[2];

    // Transpose of forward matrix:
    // Forward:  R = Y + 0*U + 1.1398374036*V
    //           G = Y - 0.3946517044*U - 0.5805986066*V
    //           B = Y + 2.0321100920*U + 0*V
    //
    // Transpose: dY = dR + dG + dB
    //            dU = 0*dR - 0.3946517044*dG + 2.0321100920*dB
    //            dV = 1.1398374036*dR - 0.5805986066*dG + 0*dB
    grad_yuv[0] = dR + dG + dB;
    grad_yuv[1] = T(-0.3946517044) * dG + T(2.0321100920) * dB;
    grad_yuv[2] = T(1.1398374036) * dR + T(-0.5805986066) * dG;
}

}  // namespace torchscience::kernel::graphics::color
