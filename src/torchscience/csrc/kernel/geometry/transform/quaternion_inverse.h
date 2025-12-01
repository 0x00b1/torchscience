#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Inverse of a unit quaternion (scalar-first wxyz convention).
 * For unit quaternions, the inverse equals the conjugate:
 * q^(-1) = [w, -x, -y, -z]
 */
template <typename T>
void quaternion_inverse_scalar(const T* q, T* output) {
  output[0] = q[0];   // w unchanged
  output[1] = -q[1];  // -x
  output[2] = -q[2];  // -y
  output[3] = -q[3];  // -z
}

}  // namespace torchscience::kernel::geometry::transform
