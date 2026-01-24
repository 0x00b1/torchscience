#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to YCbCr (ITU-R BT.601).
 *
 * Implements BT.601 color space conversion for SDTV:
 *   Y  = 0.299*R + 0.587*G + 0.114*B
 *   Cb = -0.168736*R - 0.331264*G + 0.5*B + 0.5
 *   Cr = 0.5*R - 0.418688*G - 0.081312*B + 0.5
 *
 * Input sRGB values are assumed to be in [0, 1].
 * Output Y is in [0, 1], Cb and Cr are in [0, 1] (centered at 0.5).
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param ycbcr Output array [Y, Cb, Cr]
 */
template <typename T>
void srgb_to_ycbcr_scalar(const T* rgb, T* ycbcr) {
    const T r = rgb[0];
    const T g = rgb[1];
    const T b = rgb[2];

    // BT.601 conversion matrix with offset for Cb/Cr
    ycbcr[0] = T(0.299) * r + T(0.587) * g + T(0.114) * b;           // Y
    ycbcr[1] = T(-0.168736) * r + T(-0.331264) * g + T(0.5) * b + T(0.5);  // Cb
    ycbcr[2] = T(0.5) * r + T(-0.418688) * g + T(-0.081312) * b + T(0.5);  // Cr
}

}  // namespace torchscience::kernel::graphics::color
