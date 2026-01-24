#pragma once

#include <cmath>

#include "luv_to_srgb_backward.h"

namespace torchscience::kernel::graphics::color {

/**
 * Backward pass for LCHuv to sRGB conversion.
 *
 * Computes gradients w.r.t. LCH input given gradients w.r.t. sRGB output.
 * Uses chain rule through the composition: LCH -> LUV -> sRGB
 *
 * Derivatives for LCH -> LUV:
 * - dL_luv/dL = 1
 * - du/dC = cos(h), du/dh = -C*sin(h)
 * - dv/dC = sin(h), dv/dh = C*cos(h)
 *
 * @param grad_rgb Gradient w.r.t. sRGB output [dR, dG, dB]
 * @param lch Original LCH input [L*, C*, h]
 * @param grad_lch Output gradient w.r.t. LCH [dL, dC, dh]
 */
template <typename T>
void lchuv_to_srgb_backward_scalar(const T* grad_rgb, const T* lch, T* grad_lch) {
    const T L = lch[0];
    const T C = lch[1];
    const T h = lch[2];

    // Forward pass: compute LUV from LCH
    const T cos_h = std::cos(h);
    const T sin_h = std::sin(h);
    const T u = C * cos_h;
    const T v = C * sin_h;

    T luv[3] = {L, u, v};

    // Backprop through LUV -> sRGB
    T grad_luv[3];
    luv_to_srgb_backward_scalar(grad_rgb, luv, grad_luv);

    const T grad_L_luv = grad_luv[0];
    const T grad_u = grad_luv[1];
    const T grad_v = grad_luv[2];

    // Backprop through LCH -> LUV
    // L_luv = L => dL = dL_luv
    grad_lch[0] = grad_L_luv;

    // u = C * cos(h)
    // du/dC = cos(h)
    // du/dh = -C * sin(h)
    // grad_C from u = grad_u * cos(h)
    // grad_h from u = grad_u * (-C * sin(h))
    const T grad_C_from_u = grad_u * cos_h;
    const T grad_h_from_u = grad_u * (-C * sin_h);

    // v = C * sin(h)
    // dv/dC = sin(h)
    // dv/dh = C * cos(h)
    // grad_C from v = grad_v * sin(h)
    // grad_h from v = grad_v * (C * cos(h))
    const T grad_C_from_v = grad_v * sin_h;
    const T grad_h_from_v = grad_v * (C * cos_h);

    // Total gradients
    grad_lch[1] = grad_C_from_u + grad_C_from_v;
    grad_lch[2] = grad_h_from_u + grad_h_from_v;
}

}  // namespace torchscience::kernel::graphics::color
