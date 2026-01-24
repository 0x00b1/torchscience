#pragma once

#include <cmath>

#include "srgb_to_luv.h"

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to LCHuv.
 *
 * LCHuv is the cylindrical representation of CIELUV:
 * - L* is lightness (same as LUV)
 * - C* is chroma: sqrt(u*^2 + v*^2)
 * - h is hue angle: atan2(v*, u*) in radians
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param lch Output array [L*, C*, h] where L in [0,100], C >= 0, h in [-pi, pi]
 */
template <typename T>
void srgb_to_lchuv_scalar(const T* rgb, T* lch) {
    // First convert sRGB to LUV
    T luv[3];
    srgb_to_luv_scalar(rgb, luv);

    const T L = luv[0];
    const T u = luv[1];
    const T v = luv[2];

    // Convert LUV to LCH
    // L stays the same
    lch[0] = L;

    // C = sqrt(u^2 + v^2)
    const T C = std::sqrt(u * u + v * v);
    lch[1] = C;

    // h = atan2(v, u)
    // When C = 0, hue is undefined; we set it to 0
    T h;
    if (C < T(1e-10)) {
        h = T(0);
    } else {
        h = std::atan2(v, u);
    }
    lch[2] = h;
}

}  // namespace torchscience::kernel::graphics::color
