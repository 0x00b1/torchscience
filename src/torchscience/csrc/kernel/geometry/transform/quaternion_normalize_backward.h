#pragma once

#include <cmath>

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion normalization.
 *
 * Let n = ||q||, then output = q/n
 * d(q/n)/dq_i = (n*I - q*q^T/n) / n^2 = (I - q_hat*q_hat^T) / n
 * where q_hat = q/n
 *
 * For the chain rule with grad_output:
 * grad_q = (grad_output - (grad_output . q_hat) * q_hat) / ||q||
 */
template <typename T>
void quaternion_normalize_backward_scalar(const T* grad_output, const T* q, T* grad_q) {
  const T norm = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  const T inv_norm = T(1) / norm;

  // q_hat = q / ||q||
  const T q_hat[4] = {q[0] * inv_norm, q[1] * inv_norm, q[2] * inv_norm, q[3] * inv_norm};

  // dot = grad_output . q_hat
  const T dot = grad_output[0]*q_hat[0] + grad_output[1]*q_hat[1] +
                grad_output[2]*q_hat[2] + grad_output[3]*q_hat[3];

  // grad_q = (grad_output - dot * q_hat) / ||q||
  grad_q[0] = (grad_output[0] - dot * q_hat[0]) * inv_norm;
  grad_q[1] = (grad_output[1] - dot * q_hat[1]) * inv_norm;
  grad_q[2] = (grad_output[2] - dot * q_hat[2]) * inv_norm;
  grad_q[3] = (grad_output[3] - dot * q_hat[3]) * inv_norm;
}

}  // namespace torchscience::kernel::geometry::transform
