#pragma once

#include <cmath>

#include "luv_to_srgb.h"

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single LCHuv pixel to sRGB.
 *
 * LCHuv is the cylindrical representation of CIELUV:
 * - L* is lightness (same as LUV)
 * - C* is chroma: sqrt(u*^2 + v*^2)
 * - h is hue angle in radians
 *
 * The conversion first transforms LCH to LUV:
 * - L = L
 * - u = C * cos(h)
 * - v = C * sin(h)
 *
 * Then converts LUV to sRGB.
 *
 * @param lch Input array [L*, C*, h] where L in [0,100], C >= 0, h in radians
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void lchuv_to_srgb_scalar(const T* lch, T* rgb) {
    const T L = lch[0];
    const T C = lch[1];
    const T h = lch[2];

    // Convert LCH to LUV
    // u = C * cos(h)
    // v = C * sin(h)
    T luv[3];
    luv[0] = L;
    luv[1] = C * std::cos(h);
    luv[2] = C * std::sin(h);

    // Convert LUV to sRGB
    luv_to_srgb_scalar(luv, rgb);
}

}  // namespace torchscience::kernel::graphics::color
