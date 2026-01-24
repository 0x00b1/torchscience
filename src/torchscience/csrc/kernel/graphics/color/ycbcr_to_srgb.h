#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single YCbCr pixel to sRGB (ITU-R BT.601).
 *
 * Implements BT.601 inverse color space conversion:
 *   R = Y + 1.402*(Cr - 0.5)
 *   G = Y - 0.344136*(Cb - 0.5) - 0.714136*(Cr - 0.5)
 *   B = Y + 1.772*(Cb - 0.5)
 *
 * Input Y is in [0, 1], Cb and Cr are in [0, 1] (centered at 0.5).
 * Output sRGB values are in [0, 1].
 *
 * @param ycbcr Input array [Y, Cb, Cr]
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void ycbcr_to_srgb_scalar(const T* ycbcr, T* rgb) {
    const T y = ycbcr[0];
    const T cb = ycbcr[1] - T(0.5);  // Center Cb around 0
    const T cr = ycbcr[2] - T(0.5);  // Center Cr around 0

    // BT.601 inverse conversion matrix
    rgb[0] = y + T(1.402) * cr;                          // R
    rgb[1] = y + T(-0.344136) * cb + T(-0.714136) * cr;  // G
    rgb[2] = y + T(1.772) * cb;                          // B
}

}  // namespace torchscience::kernel::graphics::color
