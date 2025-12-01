#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Hamilton product of two quaternions (scalar-first wxyz convention).
 * q1 * q2 represents rotation q1 followed by rotation q2.
 */
template <typename T>
void quaternion_multiply_scalar(const T* q1, const T* q2, T* output) {
  const T w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const T w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];

  output[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
  output[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
  output[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
  output[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
}

}  // namespace torchscience::kernel::geometry::transform
