#pragma once

#include <cmath>

#include "oklab_to_srgb_backward.h"

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for Oklch to sRGB conversion.
 *
 * Computes gradients w.r.t. Oklch input given gradients w.r.t. sRGB output.
 * Uses chain rule through the composition: Oklch -> Oklab -> sRGB
 *
 * Derivatives for Oklch -> Oklab:
 * - dL_lab/dL = 1
 * - da/dC = cos(h), da/dh = -C*sin(h)
 * - db/dC = sin(h), db/dh = C*cos(h)
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param lch Original Oklch input [L, C, h]
 * @param grad_lch Output gradient w.r.t. Oklch [dL, dC, dh]
 */
template <typename T>
void oklch_to_srgb_backward_scalar(const T* grad_rgb, const T* lch, T* grad_lch) {
    const T L = lch[0];
    const T C = lch[1];
    const T h = lch[2];

    // Forward pass: compute Oklab from Oklch
    const T cos_h = std::cos(h);
    const T sin_h = std::sin(h);
    const T a = C * cos_h;
    const T b = C * sin_h;

    T lab[3] = {L, a, b};

    // Backprop through Oklab -> sRGB
    T grad_lab[3];
    oklab_to_srgb_backward_scalar(grad_rgb, lab, grad_lab);

    const T grad_L_lab = grad_lab[0];
    const T grad_a = grad_lab[1];
    const T grad_b = grad_lab[2];

    // Backprop through Oklch -> Oklab
    // L_lab = L => dL = dL_lab
    grad_lch[0] = grad_L_lab;

    // a = C * cos(h)
    // da/dC = cos(h)
    // da/dh = -C * sin(h)
    // grad_C from a = grad_a * cos(h)
    // grad_h from a = grad_a * (-C * sin(h))
    const T grad_C_from_a = grad_a * cos_h;
    const T grad_h_from_a = grad_a * (-C * sin_h);

    // b = C * sin(h)
    // db/dC = sin(h)
    // db/dh = C * cos(h)
    // grad_C from b = grad_b * sin(h)
    // grad_h from b = grad_b * (C * cos(h))
    const T grad_C_from_b = grad_b * sin_h;
    const T grad_h_from_b = grad_b * (C * cos_h);

    // Total gradients
    grad_lch[1] = grad_C_from_a + grad_C_from_b;
    grad_lch[2] = grad_h_from_a + grad_h_from_b;
}

}  // namespace torchscience::kernel::graphics::color
