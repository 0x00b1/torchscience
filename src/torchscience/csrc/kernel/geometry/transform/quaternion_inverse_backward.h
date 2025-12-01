#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Backward pass for quaternion inverse.
 * Since inverse = conjugate for unit quaternions:
 * d(inverse)/dq = [1, -1, -1, -1] (element-wise)
 * grad_q = grad_output * [1, -1, -1, -1]
 */
template <typename T>
void quaternion_inverse_backward_scalar(const T* grad_output, T* grad_q) {
  grad_q[0] = grad_output[0];   // d/dw = 1
  grad_q[1] = -grad_output[1];  // d/dx = -1
  grad_q[2] = -grad_output[2];  // d/dy = -1
  grad_q[3] = -grad_output[3];  // d/dz = -1
}

}  // namespace torchscience::kernel::geometry::transform
