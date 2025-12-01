#pragma once

namespace torchscience::kernel::geometry::transform {

template <typename T>
void reflect_backward_scalar(
    const T* grad_output,
    const T* direction,
    const T* normal,
    T* grad_direction,
    T* grad_normal
) {
  const T d_dot_n = direction[0] * normal[0] +
                    direction[1] * normal[1] +
                    direction[2] * normal[2];
  const T g_dot_n = grad_output[0] * normal[0] +
                    grad_output[1] * normal[1] +
                    grad_output[2] * normal[2];

  const T two_g_dot_n = T(2) * g_dot_n;
  grad_direction[0] = grad_output[0] - two_g_dot_n * normal[0];
  grad_direction[1] = grad_output[1] - two_g_dot_n * normal[1];
  grad_direction[2] = grad_output[2] - two_g_dot_n * normal[2];

  const T two_d_dot_n = T(2) * d_dot_n;
  grad_normal[0] = -(two_d_dot_n * grad_output[0] + two_g_dot_n * direction[0]);
  grad_normal[1] = -(two_d_dot_n * grad_output[1] + two_g_dot_n * direction[1]);
  grad_normal[2] = -(two_d_dot_n * grad_output[2] + two_g_dot_n * direction[2]);
}

}  // namespace torchscience::kernel::geometry::transform
