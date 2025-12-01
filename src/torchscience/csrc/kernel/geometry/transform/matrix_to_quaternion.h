#pragma once

#include <cmath>

namespace torchscience::kernel::geometry::transform {

/**
 * Convert 3x3 rotation matrix to unit quaternion using Shepperd's method.
 * Chooses largest diagonal element for numerical stability.
 *
 * @param m Rotation matrix (row-major, flattened), shape 9
 * @param q Output quaternion [w, x, y, z], shape 4
 */
template <typename T>
void matrix_to_quaternion_scalar(const T* m, T* q) {
  // Row-major indexing: m[row * 3 + col]
  const T m00 = m[0], m01 = m[1], m02 = m[2];
  const T m10 = m[3], m11 = m[4], m12 = m[5];
  const T m20 = m[6], m21 = m[7], m22 = m[8];

  const T trace = m00 + m11 + m22;

  if (trace > T(0)) {
    const T s = std::sqrt(trace + T(1)) * T(2);  // s = 4*w
    q[0] = T(0.25) * s;
    q[1] = (m21 - m12) / s;
    q[2] = (m02 - m20) / s;
    q[3] = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const T s = std::sqrt(T(1) + m00 - m11 - m22) * T(2);  // s = 4*x
    q[0] = (m21 - m12) / s;
    q[1] = T(0.25) * s;
    q[2] = (m01 + m10) / s;
    q[3] = (m02 + m20) / s;
  } else if (m11 > m22) {
    const T s = std::sqrt(T(1) + m11 - m00 - m22) * T(2);  // s = 4*y
    q[0] = (m02 - m20) / s;
    q[1] = (m01 + m10) / s;
    q[2] = T(0.25) * s;
    q[3] = (m12 + m21) / s;
  } else {
    const T s = std::sqrt(T(1) + m22 - m00 - m11) * T(2);  // s = 4*z
    q[0] = (m10 - m01) / s;
    q[1] = (m02 + m20) / s;
    q[2] = (m12 + m21) / s;
    q[3] = T(0.25) * s;
  }
}

}  // namespace torchscience::kernel::geometry::transform
