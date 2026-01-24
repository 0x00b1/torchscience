#pragma once

#include <cmath>

#include "oklab_to_srgb.h"

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single Oklch pixel to sRGB.
 *
 * Oklch is the cylindrical representation of Oklab:
 * - L is lightness (same as Oklab)
 * - C is chroma: sqrt(a^2 + b^2)
 * - h is hue angle in radians
 *
 * The conversion first transforms Oklch to Oklab:
 * - L = L
 * - a = C * cos(h)
 * - b = C * sin(h)
 *
 * Then converts Oklab to sRGB.
 *
 * @param lch Input array [L, C, h] where L in [0,1], C >= 0, h in radians
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void oklch_to_srgb_scalar(const T* lch, T* rgb) {
    const T L = lch[0];
    const T C = lch[1];
    const T h = lch[2];

    // Convert Oklch to Oklab
    // a = C * cos(h)
    // b = C * sin(h)
    T lab[3];
    lab[0] = L;
    lab[1] = C * std::cos(h);
    lab[2] = C * std::sin(h);

    // Convert Oklab to sRGB
    oklab_to_srgb_scalar(lab, rgb);
}

}  // namespace torchscience::kernel::graphics::color
