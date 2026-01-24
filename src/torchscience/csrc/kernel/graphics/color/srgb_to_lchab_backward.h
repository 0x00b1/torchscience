#pragma once

#include <cmath>

#include "srgb_to_lab.h"
#include "srgb_to_lab_backward.h"

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for sRGB to LCHab conversion.
 *
 * Computes gradients w.r.t. sRGB input given gradients w.r.t. LCH output.
 * Uses chain rule through the composition: sRGB -> Lab -> LCH
 *
 * Derivatives for Lab -> LCH:
 * - dL/dL_lab = 1
 * - dC/da = a/C, dC/db = b/C (handle C=0 case)
 * - dh/da = -b/(a^2 + b^2), dh/db = a/(a^2 + b^2) (handle a=b=0 case)
 *
 * @param grad_lch Gradient w.r.t. LCH output [dL, dC, dh]
 * @param rgb Original sRGB input [R, G, B]
 * @param grad_rgb Output gradient w.r.t. sRGB [dR, dG, dB]
 */
template <typename T>
void srgb_to_lchab_backward_scalar(const T* grad_lch, const T* rgb, T* grad_rgb) {
    // Forward pass: compute Lab from sRGB
    T lab[3];
    srgb_to_lab_scalar(rgb, lab);

    const T a = lab[1];
    const T b = lab[2];

    // Compute C = sqrt(a^2 + b^2)
    const T C_squared = a * a + b * b;
    const T C = std::sqrt(C_squared);

    // Gradients w.r.t. LCH
    const T grad_L = grad_lch[0];
    const T grad_C = grad_lch[1];
    const T grad_h = grad_lch[2];

    // Backprop through LCH -> Lab
    // L_lch = L_lab => dL_lab = dL_lch
    T grad_L_lab = grad_L;

    // C = sqrt(a^2 + b^2)
    // dC/da = a/C, dC/db = b/C
    // grad_a from C = grad_C * (a/C)
    // grad_b from C = grad_C * (b/C)
    T grad_a_from_C, grad_b_from_C;
    if (C < T(1e-10)) {
        // At C=0, the gradient is undefined/zero
        grad_a_from_C = T(0);
        grad_b_from_C = T(0);
    } else {
        grad_a_from_C = grad_C * (a / C);
        grad_b_from_C = grad_C * (b / C);
    }

    // h = atan2(b, a)
    // dh/da = -b / (a^2 + b^2)
    // dh/db = a / (a^2 + b^2)
    T grad_a_from_h, grad_b_from_h;
    if (C_squared < T(1e-20)) {
        // At a=b=0, the gradient is undefined/zero
        grad_a_from_h = T(0);
        grad_b_from_h = T(0);
    } else {
        grad_a_from_h = grad_h * (-b / C_squared);
        grad_b_from_h = grad_h * (a / C_squared);
    }

    // Total gradient w.r.t. Lab
    T grad_a_lab = grad_a_from_C + grad_a_from_h;
    T grad_b_lab = grad_b_from_C + grad_b_from_h;

    // Create gradient array for Lab
    T grad_lab[3] = {grad_L_lab, grad_a_lab, grad_b_lab};

    // Backprop through Lab -> sRGB
    srgb_to_lab_backward_scalar(grad_lab, rgb, grad_rgb);
}

}  // namespace torchscience::kernel::graphics::color
