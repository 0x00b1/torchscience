#pragma once

namespace torchscience::kernel::graphics::color {

/**
 * Convert a single YUV pixel to sRGB (BT.601 analog).
 *
 * Implements BT.601 inverse color space conversion:
 *   R = Y + 1.1398374036*V
 *   G = Y - 0.3946517044*U - 0.5805986066*V
 *   B = Y + 2.0321100920*U
 *
 * Input Y is in [0, 1], U is approximately [-0.436, 0.436],
 * V is approximately [-0.615, 0.615].
 * Output sRGB values are in [0, 1].
 *
 * Note: Unlike YCbCr, U and V are already centered around 0.
 *
 * @param yuv Input array [Y, U, V]
 * @param rgb Output array [R, G, B] in sRGB
 */
template <typename T>
void yuv_to_srgb_scalar(const T* yuv, T* rgb) {
    const T y = yuv[0];
    const T u = yuv[1];  // Already centered around 0
    const T v = yuv[2];  // Already centered around 0

    // BT.601 inverse conversion matrix
    // Exact inverse of forward matrix with coefficients derived from
    // Wr=0.299, Wg=0.587, Wb=0.114, Umax=0.436, Vmax=0.615
    rgb[0] = y + T(1.1398374036) * v;                                // R
    rgb[1] = y + T(-0.3946517044) * u + T(-0.5805986066) * v;        // G
    rgb[2] = y + T(2.0321100920) * u;                                // B
}

}  // namespace torchscience::kernel::graphics::color
