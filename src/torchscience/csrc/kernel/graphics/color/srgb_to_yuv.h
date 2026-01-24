#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single sRGB pixel to YUV (BT.601 analog).
 *
 * Implements BT.601 color space conversion for analog video:
 *   Y = 0.299*R + 0.587*G + 0.114*B
 *   U = -0.1471376975*R - 0.2888623025*G + 0.436*B
 *   V = 0.615*R - 0.5149857347*G - 0.1000142653*B
 *
 * Input sRGB values are assumed to be in [0, 1].
 * Output Y is in [0, 1], U is approximately [-0.436, 0.436],
 * V is approximately [-0.615, 0.615].
 *
 * Note: Unlike YCbCr, U and V are centered around 0 (no +0.5 offset).
 *
 * @param rgb Input array [R, G, B] in sRGB
 * @param yuv Output array [Y, U, V]
 */
template <typename T>
void srgb_to_yuv_scalar(const T* rgb, T* yuv) {
    const T r = rgb[0];
    const T g = rgb[1];
    const T b = rgb[2];

    // BT.601 conversion matrix (no offset for U/V)
    // Exact coefficients derived from Wr=0.299, Wg=0.587, Wb=0.114
    // with Umax=0.436, Vmax=0.615
    yuv[0] = T(0.299) * r + T(0.587) * g + T(0.114) * b;                     // Y
    yuv[1] = T(-0.1471376975) * r + T(-0.2888623025) * g + T(0.436) * b;     // U
    yuv[2] = T(0.615) * r + T(-0.5149857347) * g + T(-0.1000142653) * b;     // V
}

}  // namespace torchscience::kernel::graphics::color
