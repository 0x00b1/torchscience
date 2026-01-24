#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB value to linear sRGB (inline helper).
 *
 * Implements the IEC 61966-2-1 standard sRGB to linear conversion.
 */
template <typename T>
T srgb_to_linear_value(T srgb) {
    const T threshold = T(0.04045);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T gamma = T(2.4);

    if (srgb <= threshold) {
        return srgb / linear_slope;
    } else {
        return std::pow((srgb + offset) / scale, gamma);
    }
}

/**
 * Convert a single sRGB pixel to YPbPr (BT.601 analog component).
 *
 * First linearizes sRGB (gamma decode), then applies BT.601 matrix:
 *   Y  =  0.299*R_lin + 0.587*G_lin + 0.114*B_lin
 *   Pb = -0.168736*R_lin - 0.331264*G_lin + 0.5*B_lin
 *   Pr =  0.5*R_lin - 0.418688*G_lin - 0.081312*B_lin
 *
 * Input sRGB values are assumed to be in [0, 1].
 * Output Y is in [0, 1], Pb and Pr are in [-0.5, 0.5].
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param ypbpr Output array [Y, Pb, Pr]
 */
template <typename T>
void srgb_to_ypbpr_scalar(const T* rgb, T* ypbpr) {
    // First linearize sRGB (gamma decode)
    const T r_lin = srgb_to_linear_value(rgb[0]);
    const T g_lin = srgb_to_linear_value(rgb[1]);
    const T b_lin = srgb_to_linear_value(rgb[2]);

    // BT.601 conversion matrix (no offset on Pb/Pr, applied to linear RGB)
    ypbpr[0] = T(0.299) * r_lin + T(0.587) * g_lin + T(0.114) * b_lin;           // Y
    ypbpr[1] = T(-0.168736) * r_lin + T(-0.331264) * g_lin + T(0.5) * b_lin;     // Pb
    ypbpr[2] = T(0.5) * r_lin + T(-0.418688) * g_lin + T(-0.081312) * b_lin;     // Pr
}

}  // namespace torchscience::kernel::graphics::color
