#pragma once

#include <cmath>

#include "srgb_to_luv.h"
#include "srgb_to_luv_backward.h"

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to LCHuv conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. LCH output.
 * Uses chain rule through the composition: sRGB -> LUV -> LCH
 *
 * Derivatives for LUV -> LCH:
 * - dL/dL_luv = 1
 * - dC/du = u/C, dC/dv = v/C (handle C=0 case)
 * - dh/du = -v/(u^2 + v^2), dh/dv = u/(u^2 + v^2) (handle u=v=0 case)
 *
 * @param grad_lch Gradient w.r.t. LCH output [dL, dC, dh]
 * @param rgb Original sRGB input [R, G, B]
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_lchuv_backward_scalar(const T* grad_lch, const T* rgb, T* grad_rgb) {
    // Forward pass: compute LUV from sRGB
    T luv[3];
    srgb_to_luv_scalar(rgb, luv);

    const T u = luv[1];
    const T v = luv[2];

    // Compute C = sqrt(u^2 + v^2)
    const T C_squared = u * u + v * v;
    const T C = std::sqrt(C_squared);

    // Gradients w.r.t. LCH
    const T grad_L = grad_lch[0];
    const T grad_C = grad_lch[1];
    const T grad_h = grad_lch[2];

    // Backprop through LCH -> LUV
    // L_lch = L_luv => dL_luv = dL_lch
    T grad_L_luv = grad_L;

    // C = sqrt(u^2 + v^2)
    // dC/du = u/C, dC/dv = v/C
    // grad_u from C = grad_C * (u/C)
    // grad_v from C = grad_C * (v/C)
    T grad_u_from_C, grad_v_from_C;
    if (C < T(1e-10)) {
        // At C=0, the gradient is undefined/zero
        grad_u_from_C = T(0);
        grad_v_from_C = T(0);
    } else {
        grad_u_from_C = grad_C * (u / C);
        grad_v_from_C = grad_C * (v / C);
    }

    // h = atan2(v, u)
    // dh/du = -v / (u^2 + v^2)
    // dh/dv = u / (u^2 + v^2)
    T grad_u_from_h, grad_v_from_h;
    if (C_squared < T(1e-20)) {
        // At u=v=0, the gradient is undefined/zero
        grad_u_from_h = T(0);
        grad_v_from_h = T(0);
    } else {
        grad_u_from_h = grad_h * (-v / C_squared);
        grad_v_from_h = grad_h * (u / C_squared);
    }

    // Total gradient w.r.t. LUV
    T grad_u_luv = grad_u_from_C + grad_u_from_h;
    T grad_v_luv = grad_v_from_C + grad_v_from_h;

    // Create gradient array for LUV
    T grad_luv[3] = {grad_L_luv, grad_u_luv, grad_v_luv};

    // Backprop through LUV -> sRGB
    srgb_to_luv_backward_scalar(grad_luv, rgb, grad_rgb);
}

}  // namespace torchscience::kernel::graphics::color
