#pragma once

#include <cmath>

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single linear RGB value to sRGB (inline helper).
 *
 * Implements the IEC 61966-2-1 standard linear to sRGB conversion.
 */
template <typename T>
T linear_to_srgb_value(T linear) {
    const T threshold = T(0.0031308);
    const T linear_slope = T(12.92);
    const T offset = T(0.055);
    const T scale = T(1.055);
    const T inverse_gamma = T(1.0 / 2.4);

    if (linear <= threshold) {
        return linear * linear_slope;
    } else {
        return scale * std::pow(linear, inverse_gamma) - offset;
    }
}

/**
 * Convert a single YPbPr pixel to sRGB (BT.601 inverse).
 *
 * Applies BT.601 inverse matrix, then gamma encodes:
 *   R_lin = Y + 1.402*Pr
 *   G_lin = Y - 0.344136*Pb - 0.714136*Pr
 *   B_lin = Y + 1.772*Pb
 *
 * Input Y is in [0, 1], Pb and Pr are in [-0.5, 0.5].
 * Output sRGB values are in [0, 1].
 *
 * @param ypbpr Input array [Y, Pb, Pr]
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void ypbpr_to_srgb_scalar(const T* ypbpr, T* rgb) {
    const T y = ypbpr[0];
    const T pb = ypbpr[1];  // Already centered at 0 (no offset subtraction needed)
    const T pr = ypbpr[2];  // Already centered at 0

    // BT.601 inverse conversion matrix to linear RGB
    const T r_lin = y + T(1.402) * pr;
    const T g_lin = y + T(-0.344136) * pb + T(-0.714136) * pr;
    const T b_lin = y + T(1.772) * pb;

    // Apply gamma encoding (linear to sRGB)
    rgb[0] = linear_to_srgb_value(r_lin);
    rgb[1] = linear_to_srgb_value(g_lin);
    rgb[2] = linear_to_srgb_value(b_lin);
}

}  // namespace torchscience::kernel::graphics::color
