#pragma once

#include <cmath>

#include "srgb_to_oklab.h"

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to Oklch.
 *
 * Oklch is the cylindrical representation of Oklab:
 * - L is lightness (same as Oklab)
 * - C is chroma: sqrt(a^2 + b^2)
 * - h is hue angle: atan2(b, a) in radians
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param lch Output array [L, C, h] where L in [0,1], C >= 0, h in [-pi, pi]
 */
template <typename T>
void srgb_to_oklch_scalar(const T* rgb, T* lch) {
    // First convert sRGB to Oklab
    T lab[3];
    srgb_to_oklab_scalar(rgb, lab);

    const T L = lab[0];
    const T a = lab[1];
    const T b = lab[2];

    // Convert Oklab to Oklch
    // L stays the same
    lch[0] = L;

    // C = sqrt(a^2 + b^2)
    const T C = std::sqrt(a * a + b * b);
    lch[1] = C;

    // h = atan2(b, a)
    // When C = 0, hue is undefined; we set it to 0
    T h;
    if (C < T(1e-10)) {
        h = T(0);
    } else {
        h = std::atan2(b, a);
    }
    lch[2] = h;
}

}  // namespace torchscience::kernel::graphics::color
