#pragma once

#include <cmath>

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion_slerp.
 *
 * Computes gradients with respect to q1, q2, and t.
 *
 * The forward pass computes:
 *   pre_norm = s1 * q1 + s2 * q2_adj
 *   output = pre_norm / ||pre_norm||
 *
 * where s1 = sin((1-t)*theta)/sin(theta), s2 = sin(t*theta)/sin(theta)
 * and theta = acos(dot(q1, q2_adj)).
 */
template <typename T>
void quaternion_slerp_backward_scalar(const T* grad_output,
                                      const T* q1,
                                      const T* q2,
                                      T t,
                                      T* grad_q1,
                                      T* grad_q2,
                                      T* grad_t  // scalar output
) {
  // Compute dot product
  T dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];

  // Adjust q2 if needed
  T sign = T(1);
  if (dot < T(0)) {
    dot = -dot;
    sign = T(-1);
  }
  if (dot > T(1))
    dot = T(1);

  T s1, s2;
  T ds1_dt, ds2_dt;
  T ds1_dtheta, ds2_dtheta;
  T theta;

  if (dot > T(0.9995)) {
    // Linear interpolation case
    s1 = T(1) - t;
    s2 = t;
    ds1_dt = T(-1);
    ds2_dt = T(1);
    ds1_dtheta = T(0);
    ds2_dtheta = T(0);
    theta = T(0);
  } else {
    theta = std::acos(dot);
    const T sin_theta = std::sin(theta);
    const T inv_sin_theta = T(1) / sin_theta;
    const T cos_theta = dot;

    const T sin_1mt_theta = std::sin((T(1) - t) * theta);
    const T sin_t_theta = std::sin(t * theta);
    const T cos_1mt_theta = std::cos((T(1) - t) * theta);
    const T cos_t_theta = std::cos(t * theta);

    s1 = sin_1mt_theta * inv_sin_theta;
    s2 = sin_t_theta * inv_sin_theta;

    // Derivatives w.r.t. t
    ds1_dt = -theta * cos_1mt_theta * inv_sin_theta;
    ds2_dt = theta * cos_t_theta * inv_sin_theta;

    // Derivatives w.r.t. theta
    ds1_dtheta = (T(1) - t) * cos_1mt_theta * inv_sin_theta -
                 sin_1mt_theta * cos_theta * inv_sin_theta * inv_sin_theta;
    ds2_dtheta = t * cos_t_theta * inv_sin_theta -
                 sin_t_theta * cos_theta * inv_sin_theta * inv_sin_theta;
  }

  // Compute pre_norm = s1 * q1 + s2 * q2_adj
  T pre_norm[4];
  for (int i = 0; i < 4; ++i) {
    pre_norm[i] = s1 * q1[i] + s2 * sign * q2[i];
  }

  // Compute norm of pre_norm
  T norm_sq = pre_norm[0] * pre_norm[0] + pre_norm[1] * pre_norm[1] +
              pre_norm[2] * pre_norm[2] + pre_norm[3] * pre_norm[3];
  T norm = std::sqrt(norm_sq);
  T inv_norm = T(1) / (norm + T(1e-12));

  // The output is normalized: output = pre_norm / norm
  // Gradient through normalization: d(v/||v||)/dv = (I - vv^T/||v||^2) / ||v||
  // So: grad_pre_norm = (grad_output - output * (grad_output . output)) / norm
  // where output = pre_norm / norm

  T output[4];
  for (int i = 0; i < 4; ++i) {
    output[i] = pre_norm[i] * inv_norm;
  }

  T grad_dot_output = T(0);
  for (int i = 0; i < 4; ++i) {
    grad_dot_output += grad_output[i] * output[i];
  }

  T grad_pre_norm[4];
  for (int i = 0; i < 4; ++i) {
    grad_pre_norm[i] = (grad_output[i] - output[i] * grad_dot_output) * inv_norm;
  }

  // Now compute gradients w.r.t. q1, q2, t through pre_norm
  // pre_norm = s1 * q1 + s2 * q2_adj
  // where s1, s2 depend on theta = acos(dot(q1, q2_adj))

  // grad_t = sum_i grad_pre_norm[i] * (ds1_dt * q1[i] + ds2_dt * q2_adj[i])
  *grad_t = T(0);
  for (int i = 0; i < 4; ++i) {
    *grad_t += grad_pre_norm[i] * (ds1_dt * q1[i] + ds2_dt * sign * q2[i]);
  }

  if (dot > T(0.9995)) {
    // Linear case: s1, s2 don't depend on q1, q2
    // grad_q1 = s1 * grad_pre_norm
    // grad_q2 = s2 * sign * grad_pre_norm
    for (int i = 0; i < 4; ++i) {
      grad_q1[i] = s1 * grad_pre_norm[i];
      grad_q2[i] = s2 * sign * grad_pre_norm[i];
    }
  } else {
    // d(theta)/d(dot) = -1/sqrt(1 - dot^2)
    const T sqrt_1_minus_dot2 = std::sqrt(T(1) - dot * dot + T(1e-12));
    const T dtheta_ddot = T(-1) / sqrt_1_minus_dot2;

    // Compute d(pre_norm)/d(theta) = ds1_dtheta * q1 + ds2_dtheta * q2_adj
    T dprenorm_dtheta[4];
    for (int i = 0; i < 4; ++i) {
      dprenorm_dtheta[i] = ds1_dtheta * q1[i] + ds2_dtheta * sign * q2[i];
    }

    // grad_pre_norm^T * d(pre_norm)/d(theta)
    T grad_theta = T(0);
    for (int i = 0; i < 4; ++i) {
      grad_theta += grad_pre_norm[i] * dprenorm_dtheta[i];
    }

    // grad_dot = grad_theta * dtheta_ddot
    T grad_dot = grad_theta * dtheta_ddot;

    // d(dot)/d(q1) = sign * q2
    // d(dot)/d(q2) = sign * q1
    for (int i = 0; i < 4; ++i) {
      // Direct contribution from pre_norm = s1 * q1 + s2 * q2_adj
      grad_q1[i] = s1 * grad_pre_norm[i];
      grad_q2[i] = s2 * sign * grad_pre_norm[i];

      // Contribution through theta via dot product
      grad_q1[i] += grad_dot * sign * q2[i];
      grad_q2[i] += grad_dot * sign * q1[i];
    }
  }
}

}  // namespace torchscience::kernel::geometry::transform
