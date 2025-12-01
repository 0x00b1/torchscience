#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion_to_matrix.
 * Computes gradient w.r.t. quaternion q from gradient of matrix.
 *
 * The matrix elements (row-major) are:
 *   M[0] = 1 - 2y^2 - 2z^2    M[1] = 2xy - 2wz         M[2] = 2xz + 2wy
 *   M[3] = 2xy + 2wz          M[4] = 1 - 2x^2 - 2z^2   M[5] = 2yz - 2wx
 *   M[6] = 2xz - 2wy          M[7] = 2yz + 2wx         M[8] = 1 - 2x^2 - 2y^2
 *
 * Gradients derived from the chain rule:
 *   dL/dq_i = sum_j (dL/dM_j * dM_j/dq_i)
 *
 * @param grad_output Gradient w.r.t. matrix output (row-major), shape 9
 * @param q Input quaternion [w, x, y, z], shape 4
 * @param grad_q Output gradient w.r.t. quaternion, shape 4
 */
template <typename T>
void quaternion_to_matrix_backward_scalar(const T* grad_output, const T* q, T* grad_q) {
  const T w = q[0], x = q[1], y = q[2], z = q[3];

  // grad_output has shape 9 (flattened 3x3 row-major)
  const T g0 = grad_output[0], g1 = grad_output[1], g2 = grad_output[2];
  const T g3 = grad_output[3], g4 = grad_output[4], g5 = grad_output[5];
  const T g6 = grad_output[6], g7 = grad_output[7], g8 = grad_output[8];

  // Derivatives of matrix elements w.r.t. w, x, y, z:
  //
  // dM[0]/dw = 0,  dM[1]/dw = -2z, dM[2]/dw = 2y,
  // dM[3]/dw = 2z, dM[4]/dw = 0,   dM[5]/dw = -2x,
  // dM[6]/dw = -2y, dM[7]/dw = 2x, dM[8]/dw = 0
  //
  // dM[0]/dx = 0,   dM[1]/dx = 2y,  dM[2]/dx = 2z,
  // dM[3]/dx = 2y,  dM[4]/dx = -4x, dM[5]/dx = -2w,
  // dM[6]/dx = 2z,  dM[7]/dx = 2w,  dM[8]/dx = -4x
  //
  // dM[0]/dy = -4y, dM[1]/dy = 2x,  dM[2]/dy = 2w,
  // dM[3]/dy = 2x,  dM[4]/dy = 0,   dM[5]/dy = 2z,
  // dM[6]/dy = -2w, dM[7]/dy = 2z,  dM[8]/dy = -4y
  //
  // dM[0]/dz = -4z, dM[1]/dz = -2w, dM[2]/dz = 2x,
  // dM[3]/dz = 2w,  dM[4]/dz = -4z, dM[5]/dz = 2y,
  // dM[6]/dz = 2x,  dM[7]/dz = 2y,  dM[8]/dz = 0

  // dL/dw = 2*(-z*g1 + y*g2 + z*g3 - x*g5 - y*g6 + x*g7)
  grad_q[0] = T(2) * (-z*g1 + y*g2 + z*g3 - x*g5 - y*g6 + x*g7);

  // dL/dx = 2*(y*g1 + z*g2 + y*g3 - 2x*g4 - w*g5 + z*g6 + w*g7 - 2x*g8)
  grad_q[1] = T(2) * (y*g1 + z*g2 + y*g3 - T(2)*x*g4 - w*g5 + z*g6 + w*g7 - T(2)*x*g8);

  // dL/dy = 2*(-2y*g0 + x*g1 + w*g2 + x*g3 + z*g5 - w*g6 + z*g7 - 2y*g8)
  grad_q[2] = T(2) * (-T(2)*y*g0 + x*g1 + w*g2 + x*g3 + z*g5 - w*g6 + z*g7 - T(2)*y*g8);

  // dL/dz = 2*(-2z*g0 - w*g1 + x*g2 + w*g3 - 2z*g4 + y*g5 + x*g6 + y*g7)
  grad_q[3] = T(2) * (-T(2)*z*g0 - w*g1 + x*g2 + w*g3 - T(2)*z*g4 + y*g5 + x*g6 + y*g7);
}

}  // namespace torchscience::kernel::geometry::transform
