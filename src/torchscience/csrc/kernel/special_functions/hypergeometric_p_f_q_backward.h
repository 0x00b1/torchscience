#pragma once

#include <cmath>
#include <vector>

#include "hypergeometric_p_f_q.h"

namespace torchscience::kernel::special_functions {

// Backward pass for generalized hypergeometric pFq function
// Computes gradients with respect to each element of a[], b[], and z
//
// grad_a[i] = grad * ∂f/∂a[i]
// grad_b[j] = grad * ∂f/∂b[j]
// grad_z = grad * ∂f/∂z
//
// The derivative with respect to z has a nice closed form:
// ∂pFq/∂z = [∏_i a[i] / ∏_j b[j]] * pFq(a+1; b+1; z)
//
// For parameter derivatives, we use finite differences.
template <typename T>
void hypergeometric_p_f_q_backward(
    T grad,
    const T* a, int p,
    const T* b, int q,
    T z,
    T* grad_a,  // Output: gradients for a (length p)
    T* grad_b,  // Output: gradients for b (length q)
    T& grad_z   // Output: gradient for z
) {
  using detail::pfq_epsilon;
  using detail::pfq_is_complex_v;
  using detail::pfq_real_type_t;

  // Compute ∂f/∂z = [∏_i a[i] / ∏_j b[j]] * pFq(a+1; b+1; z)
  // First compute the coefficient
  T coeff = T(1);
  for (int i = 0; i < p; ++i) {
    coeff *= a[i];
  }
  for (int j = 0; j < q; ++j) {
    if (std::abs(b[j]) > pfq_epsilon<T>()) {
      coeff /= b[j];
    } else {
      coeff = T(0);  // Avoid division by zero
    }
  }

  // Create shifted parameter arrays
  std::vector<T> a_plus1(p);
  std::vector<T> b_plus1(q);
  for (int i = 0; i < p; ++i) {
    a_plus1[i] = a[i] + T(1);
  }
  for (int j = 0; j < q; ++j) {
    b_plus1[j] = b[j] + T(1);
  }

  // Evaluate pFq with shifted parameters
  T f_shifted = hypergeometric_p_f_q(a_plus1.data(), p, b_plus1.data(), q, z);
  T dfdz = coeff * f_shifted;

  // For parameter derivatives, use finite differences
  using real_t = pfq_real_type_t<T>;
  real_t eps_real = std::sqrt(pfq_epsilon<T>());
  T eps = T(eps_real);

  // Compute ∂f/∂a[i] for each i
  std::vector<T> a_work(p);
  for (int i = 0; i < p; ++i) {
    // Copy a to work array
    for (int k = 0; k < p; ++k) {
      a_work[k] = a[k];
    }

    // f(a[i] + eps)
    a_work[i] = a[i] + eps;
    T f_plus = hypergeometric_p_f_q(a_work.data(), p, b, q, z);

    // f(a[i] - eps)
    a_work[i] = a[i] - eps;
    T f_minus = hypergeometric_p_f_q(a_work.data(), p, b, q, z);

    T dfda_i = (f_plus - f_minus) / (T(2) * eps);

    // Apply chain rule
    if constexpr (pfq_is_complex_v<T>) {
      grad_a[i] = grad * std::conj(dfda_i);
    } else {
      grad_a[i] = grad * dfda_i;
    }
  }

  // Compute ∂f/∂b[j] for each j
  std::vector<T> b_work(q);
  for (int j = 0; j < q; ++j) {
    // Copy b to work array
    for (int k = 0; k < q; ++k) {
      b_work[k] = b[k];
    }

    // f(b[j] + eps)
    b_work[j] = b[j] + eps;
    T f_plus = hypergeometric_p_f_q(a, p, b_work.data(), q, z);

    // f(b[j] - eps)
    b_work[j] = b[j] - eps;
    T f_minus = hypergeometric_p_f_q(a, p, b_work.data(), q, z);

    T dfdb_j = (f_plus - f_minus) / (T(2) * eps);

    // Apply chain rule
    if constexpr (pfq_is_complex_v<T>) {
      grad_b[j] = grad * std::conj(dfdb_j);
    } else {
      grad_b[j] = grad * dfdb_j;
    }
  }

  // Apply chain rule for z
  if constexpr (pfq_is_complex_v<T>) {
    grad_z = grad * std::conj(dfdz);
  } else {
    grad_z = grad * dfdz;
  }
}

} // namespace torchscience::kernel::special_functions
