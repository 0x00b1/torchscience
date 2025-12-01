#pragma once

namespace torchscience::kernel::geometry::transform {

/**
 * Scalar kernel for Snell's law refraction backward pass.
 *
 * Forward: T = eta * D + (eta * cos_i - cos_t) * N
 * where:
 *   - cos_i = -D.N
 *   - cos_t = sqrt(1 - eta^2 * (1 - cos_i^2))
 *
 * Gradients:
 *   dL/dD = dL/dT * dT/dD
 *   dL/dN = dL/dT * dT/dN
 *   dL/deta = dL/dT * dT/deta
 *
 * For TIR case (output was zero), gradients are zero.
 */
template <typename T>
void refract_backward_scalar(
    const T* grad_output,
    const T* direction,
    const T* normal,
    T eta,
    T* grad_direction,
    T* grad_normal,
    T* grad_eta
) {
  // Recompute forward values
  const T cos_i = -(direction[0] * normal[0] +
                    direction[1] * normal[1] +
                    direction[2] * normal[2]);
  const T sin2_i = T(1) - cos_i * cos_i;
  const T sin2_t = eta * eta * sin2_i;

  // Check for TIR - gradients are zero
  if (sin2_t > T(1)) {
    grad_direction[0] = T(0);
    grad_direction[1] = T(0);
    grad_direction[2] = T(0);
    grad_normal[0] = T(0);
    grad_normal[1] = T(0);
    grad_normal[2] = T(0);
    *grad_eta = T(0);
    return;
  }

  const T cos_t = std::sqrt(T(1) - sin2_t);
  const T k = eta * cos_i - cos_t;

  // Compute grad_output dot products
  const T g_dot_n = grad_output[0] * normal[0] +
                    grad_output[1] * normal[1] +
                    grad_output[2] * normal[2];
  const T g_dot_d = grad_output[0] * direction[0] +
                    grad_output[1] * direction[1] +
                    grad_output[2] * direction[2];

  // Derivative of cos_t w.r.t. cos_i:
  // cos_t = sqrt(1 - eta^2 * (1 - cos_i^2))
  // d(cos_t)/d(cos_i) = eta^2 * cos_i / cos_t
  const T dcos_t_dcos_i = (cos_t > T(1e-8)) ? (eta * eta * cos_i / cos_t) : T(0);

  // Derivative of k w.r.t. cos_i:
  // k = eta * cos_i - cos_t
  // dk/d(cos_i) = eta - d(cos_t)/d(cos_i) = eta - eta^2 * cos_i / cos_t
  const T dk_dcos_i = eta - dcos_t_dcos_i;

  // T = eta * D + k * N
  // dT/dD = eta * I + dk/dcos_i * d(cos_i)/dD * N^T
  //       = eta * I + dk/dcos_i * (-N) * N^T
  // dL/dD = eta * grad_output + dk/dcos_i * (-g_dot_n) * N
  //       = eta * grad_output - dk/dcos_i * g_dot_n * N
  const T c1 = -dk_dcos_i * g_dot_n;
  grad_direction[0] = eta * grad_output[0] + c1 * normal[0];
  grad_direction[1] = eta * grad_output[1] + c1 * normal[1];
  grad_direction[2] = eta * grad_output[2] + c1 * normal[2];

  // dT/dN = k * I + dk/dcos_i * d(cos_i)/dN * N^T
  //       = k * I + dk/dcos_i * (-D) * N^T
  // But wait, d(cos_i)/dN = -D, and the outer product is with grad_output...
  // dL/dN = k * grad_output + dk/dcos_i * d(cos_i)/dN * (grad_output.N)
  //       = k * grad_output + dk/dcos_i * (-D) * g_dot_n
  //       = k * grad_output - dk/dcos_i * g_dot_n * D
  const T c2 = -dk_dcos_i * g_dot_n;
  grad_normal[0] = k * grad_output[0] + c2 * direction[0];
  grad_normal[1] = k * grad_output[1] + c2 * direction[1];
  grad_normal[2] = k * grad_output[2] + c2 * direction[2];

  // Derivative of k w.r.t. eta:
  // k = eta * cos_i - cos_t
  // cos_t = sqrt(1 - eta^2 * sin2_i)
  // d(cos_t)/d(eta) = -eta * sin2_i / cos_t
  // dk/d(eta) = cos_i + eta * sin2_i / cos_t
  const T dcos_t_deta = (cos_t > T(1e-8)) ? (-eta * sin2_i / cos_t) : T(0);
  const T dk_deta = cos_i - dcos_t_deta;

  // dT/deta = D + dk/deta * N
  // dL/deta = grad_output . (D + dk/deta * N)
  //         = g_dot_d + dk_deta * g_dot_n
  *grad_eta = g_dot_d + dk_deta * g_dot_n;
}

}  // namespace torchscience::kernel::geometry::transform
