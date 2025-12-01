#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion multiplication.
 * Computes gradients w.r.t. q1 and q2 given gradient of output.
 */
template <typename T>
void quaternion_multiply_backward_scalar(
    const T* grad_output,
    const T* q1,
    const T* q2,
    T* grad_q1,
    T* grad_q2
) {
  const T w1 = q1[0], x1 = q1[1], y1 = q1[2], z1 = q1[3];
  const T w2 = q2[0], x2 = q2[1], y2 = q2[2], z2 = q2[3];
  const T gw = grad_output[0], gx = grad_output[1];
  const T gy = grad_output[2], gz = grad_output[3];

  // Gradient w.r.t. q1
  // d(output)/d(w1) = [w2, x2, y2, z2]
  // d(output)/d(x1) = [-x2, w2, -z2, y2]
  // d(output)/d(y1) = [-y2, z2, w2, -x2]
  // d(output)/d(z1) = [-z2, -y2, x2, w2]
  grad_q1[0] = gw * w2 + gx * x2 + gy * y2 + gz * z2;
  grad_q1[1] = -gw * x2 + gx * w2 - gy * z2 + gz * y2;
  grad_q1[2] = -gw * y2 + gx * z2 + gy * w2 - gz * x2;
  grad_q1[3] = -gw * z2 - gx * y2 + gy * x2 + gz * w2;

  // Gradient w.r.t. q2
  // d(output)/d(w2) = [w1, x1, y1, z1]
  // d(output)/d(x2) = [-x1, w1, z1, -y1]
  // d(output)/d(y2) = [-y1, -z1, w1, x1]
  // d(output)/d(z2) = [-z1, y1, -x1, w1]
  grad_q2[0] = gw * w1 + gx * x1 + gy * y1 + gz * z1;
  grad_q2[1] = -gw * x1 + gx * w1 + gy * z1 - gz * y1;
  grad_q2[2] = -gw * y1 - gx * z1 + gy * w1 + gz * x1;
  grad_q2[3] = -gw * z1 + gx * y1 - gy * x1 + gz * w1;
}

}  // namespace torchscience::kernel::geometry::transform
