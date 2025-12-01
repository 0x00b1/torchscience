#pragma once

#include <cmath>

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for matrix_to_quaternion.
 * Computes gradient w.r.t. rotation matrix from gradient of quaternion.
 *
 * This implementation handles all 4 branches of Shepperd's method.
 *
 * @param grad_output Gradient w.r.t. quaternion output [gw, gx, gy, gz], shape 4
 * @param m Input rotation matrix (row-major, flattened), shape 9
 * @param grad_m Output gradient w.r.t. matrix, shape 9
 */
template <typename T>
void matrix_to_quaternion_backward_scalar(const T* grad_output, const T* m, T* grad_m) {
  // Row-major indexing: m[row * 3 + col]
  const T m00 = m[0], m01 = m[1], m02 = m[2];
  const T m10 = m[3], m11 = m[4], m12 = m[5];
  const T m20 = m[6], m21 = m[7], m22 = m[8];

  const T gw = grad_output[0], gx = grad_output[1];
  const T gy = grad_output[2], gz = grad_output[3];

  const T trace = m00 + m11 + m22;

  // Initialize gradients to zero
  for (int i = 0; i < 9; ++i) grad_m[i] = T(0);

  if (trace > T(0)) {
    // Branch 1: trace > 0
    // s = 2 * sqrt(trace + 1) = 4*w
    // q[0] = 0.25 * s = 0.5 * sqrt(trace + 1)
    // q[1] = (m21 - m12) / s
    // q[2] = (m02 - m20) / s
    // q[3] = (m10 - m01) / s

    const T r = std::sqrt(trace + T(1));
    const T s = T(2) * r;
    const T inv_s = T(1) / s;
    const T inv_s_sq = inv_s * inv_s;

    // dq0/dtrace = 0.5 * 0.5 / sqrt(trace + 1) = 0.25 / r
    const T dq0_dtrace = T(0.25) / r;

    // For q1, q2, q3: d/dtrace of (X/s) = -X * ds/dtrace / s^2 = -X / (s^2 * r)
    // since ds/dtrace = 1/r
    const T ds_dtrace = T(1) / r;
    const T q1_val = (m21 - m12);
    const T q2_val = (m02 - m20);
    const T q3_val = (m10 - m01);

    // Gradient contributions through trace (m00, m11, m22)
    const T dtrace_grad = gw * dq0_dtrace
                        - gx * q1_val * inv_s_sq * ds_dtrace
                        - gy * q2_val * inv_s_sq * ds_dtrace
                        - gz * q3_val * inv_s_sq * ds_dtrace;

    grad_m[0] += dtrace_grad;  // m00
    grad_m[4] += dtrace_grad;  // m11
    grad_m[8] += dtrace_grad;  // m22

    // Direct contributions from off-diagonal elements
    grad_m[7] += gx * inv_s;   // m21: dq1/dm21 = 1/s
    grad_m[5] -= gx * inv_s;   // m12: dq1/dm12 = -1/s
    grad_m[2] += gy * inv_s;   // m02: dq2/dm02 = 1/s
    grad_m[6] -= gy * inv_s;   // m20: dq2/dm20 = -1/s
    grad_m[3] += gz * inv_s;   // m10: dq3/dm10 = 1/s
    grad_m[1] -= gz * inv_s;   // m01: dq3/dm01 = -1/s

  } else if (m00 > m11 && m00 > m22) {
    // Branch 2: m00 is largest diagonal
    // s = 2 * sqrt(1 + m00 - m11 - m22) = 4*x
    // q[0] = (m21 - m12) / s
    // q[1] = 0.25 * s = 0.5 * sqrt(1 + m00 - m11 - m22)
    // q[2] = (m01 + m10) / s
    // q[3] = (m02 + m20) / s

    const T diag = T(1) + m00 - m11 - m22;
    const T r = std::sqrt(diag);
    const T s = T(2) * r;
    const T inv_s = T(1) / s;
    const T inv_s_sq = inv_s * inv_s;

    const T dq1_ddiag = T(0.25) / r;
    const T ds_ddiag = T(1) / r;

    const T q0_val = (m21 - m12);
    const T q2_val = (m01 + m10);
    const T q3_val = (m02 + m20);

    // Gradient through diag = 1 + m00 - m11 - m22
    const T ddiag_grad = gx * dq1_ddiag
                       - gw * q0_val * inv_s_sq * ds_ddiag
                       - gy * q2_val * inv_s_sq * ds_ddiag
                       - gz * q3_val * inv_s_sq * ds_ddiag;

    grad_m[0] += ddiag_grad;   // m00: ddiag/dm00 = 1
    grad_m[4] -= ddiag_grad;   // m11: ddiag/dm11 = -1
    grad_m[8] -= ddiag_grad;   // m22: ddiag/dm22 = -1

    // Direct contributions
    grad_m[7] += gw * inv_s;   // m21
    grad_m[5] -= gw * inv_s;   // m12
    grad_m[1] += gy * inv_s;   // m01
    grad_m[3] += gy * inv_s;   // m10
    grad_m[2] += gz * inv_s;   // m02
    grad_m[6] += gz * inv_s;   // m20

  } else if (m11 > m22) {
    // Branch 3: m11 is largest diagonal (and m11 > m00 or m11 == m00)
    // s = 2 * sqrt(1 + m11 - m00 - m22) = 4*y
    // q[0] = (m02 - m20) / s
    // q[1] = (m01 + m10) / s
    // q[2] = 0.25 * s = 0.5 * sqrt(1 + m11 - m00 - m22)
    // q[3] = (m12 + m21) / s

    const T diag = T(1) + m11 - m00 - m22;
    const T r = std::sqrt(diag);
    const T s = T(2) * r;
    const T inv_s = T(1) / s;
    const T inv_s_sq = inv_s * inv_s;

    const T dq2_ddiag = T(0.25) / r;
    const T ds_ddiag = T(1) / r;

    const T q0_val = (m02 - m20);
    const T q1_val = (m01 + m10);
    const T q3_val = (m12 + m21);

    // Gradient through diag = 1 + m11 - m00 - m22
    const T ddiag_grad = gy * dq2_ddiag
                       - gw * q0_val * inv_s_sq * ds_ddiag
                       - gx * q1_val * inv_s_sq * ds_ddiag
                       - gz * q3_val * inv_s_sq * ds_ddiag;

    grad_m[4] += ddiag_grad;   // m11: ddiag/dm11 = 1
    grad_m[0] -= ddiag_grad;   // m00: ddiag/dm00 = -1
    grad_m[8] -= ddiag_grad;   // m22: ddiag/dm22 = -1

    // Direct contributions
    grad_m[2] += gw * inv_s;   // m02
    grad_m[6] -= gw * inv_s;   // m20
    grad_m[1] += gx * inv_s;   // m01
    grad_m[3] += gx * inv_s;   // m10
    grad_m[5] += gz * inv_s;   // m12
    grad_m[7] += gz * inv_s;   // m21

  } else {
    // Branch 4: m22 is largest diagonal
    // s = 2 * sqrt(1 + m22 - m00 - m11) = 4*z
    // q[0] = (m10 - m01) / s
    // q[1] = (m02 + m20) / s
    // q[2] = (m12 + m21) / s
    // q[3] = 0.25 * s = 0.5 * sqrt(1 + m22 - m00 - m11)

    const T diag = T(1) + m22 - m00 - m11;
    const T r = std::sqrt(diag);
    const T s = T(2) * r;
    const T inv_s = T(1) / s;
    const T inv_s_sq = inv_s * inv_s;

    const T dq3_ddiag = T(0.25) / r;
    const T ds_ddiag = T(1) / r;

    const T q0_val = (m10 - m01);
    const T q1_val = (m02 + m20);
    const T q2_val = (m12 + m21);

    // Gradient through diag = 1 + m22 - m00 - m11
    const T ddiag_grad = gz * dq3_ddiag
                       - gw * q0_val * inv_s_sq * ds_ddiag
                       - gx * q1_val * inv_s_sq * ds_ddiag
                       - gy * q2_val * inv_s_sq * ds_ddiag;

    grad_m[8] += ddiag_grad;   // m22: ddiag/dm22 = 1
    grad_m[0] -= ddiag_grad;   // m00: ddiag/dm00 = -1
    grad_m[4] -= ddiag_grad;   // m11: ddiag/dm11 = -1

    // Direct contributions
    grad_m[3] += gw * inv_s;   // m10
    grad_m[1] -= gw * inv_s;   // m01
    grad_m[2] += gx * inv_s;   // m02
    grad_m[6] += gx * inv_s;   // m20
    grad_m[5] += gy * inv_s;   // m12
    grad_m[7] += gy * inv_s;   // m21
  }
}

}  // namespace torchscience::kernel::geometry::transform
