#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Convert unit quaternion to 3x3 rotation matrix.
 *
 * Uses scalar-first (wxyz) convention: q = [w, x, y, z]
 *
 * The rotation matrix R is computed as:
 *   R = | 1 - 2(y^2 + z^2)    2(xy - wz)       2(xz + wy)   |
 *       | 2(xy + wz)          1 - 2(x^2 + z^2) 2(yz - wx)   |
 *       | 2(xz - wy)          2(yz + wx)       1 - 2(x^2 + y^2) |
 *
 * @param q Quaternion [w, x, y, z], shape 4
 * @param matrix Output rotation matrix (row-major), shape 9 (flattened 3x3)
 */
template <typename T>
void quaternion_to_matrix_scalar(const T* q, T* matrix) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];

  const T x2 = x + x, y2 = y + y, z2 = z + z;
  const T xx = x * x2, xy = x * y2, xz = x * z2;
  const T yy = y * y2, yz = y * z2, zz = z * z2;
  const T wx = w * x2, wy = w * y2, wz = w * z2;

  // Row-major: matrix[row * 3 + col]
  matrix[0] = T(1) - (yy + zz);  matrix[1] = xy - wz;           matrix[2] = xz + wy;
  matrix[3] = xy + wz;           matrix[4] = T(1) - (xx + zz);  matrix[5] = yz - wx;
  matrix[6] = xz - wy;           matrix[7] = yz + wx;           matrix[8] = T(1) - (xx + yy);
}

}  // namespace torchscience::kernel::geometry::transform
