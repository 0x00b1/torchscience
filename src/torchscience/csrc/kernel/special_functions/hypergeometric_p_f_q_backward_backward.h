#pragma once

#include <cmath>
#include <vector>

#include "hypergeometric_p_f_q.h"
#include "hypergeometric_p_f_q_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for generalized hypergeometric pFq function
// Computes gradients for the backward pass inputs
//
// Inputs:
//   gg_a[i]: gradient of gradient with respect to a[i]
//   gg_b[j]: gradient of gradient with respect to b[j]
//   gg_z: gradient of gradient with respect to z
//   grad: original upstream gradient
//   a[], b[], z: original function inputs
//
// Outputs:
//   grad_grad: gradient of output w.r.t. grad
//   grad_a[i]: gradient of output w.r.t. a[i]
//   grad_b[j]: gradient of output w.r.t. b[j]
//   grad_z: gradient of output w.r.t. z
template <typename T>
void hypergeometric_p_f_q_backward_backward(
    const T* gg_a, int p,
    const T* gg_b, int q,
    T gg_z,
    T grad,
    const T* a,
    const T* b,
    T z,
    T& grad_grad,
    T* out_grad_a,
    T* out_grad_b,
    T& out_grad_z
) {
  using detail::pfq_epsilon;
  using detail::pfq_is_complex_v;
  using detail::pfq_real_type_t;

  // Use finite differences for second-order derivatives
  using real_t = pfq_real_type_t<T>;
  real_t eps_real = std::cbrt(pfq_epsilon<T>());
  T eps = T(eps_real);

  // Get first-order derivatives at current point
  std::vector<T> da(p);
  std::vector<T> db(q);
  T dz;
  hypergeometric_p_f_q_backward(T(1), a, p, b, q, z, da.data(), db.data(), dz);

  // grad_grad = sum of gg_i * df/di
  grad_grad = gg_z * dz;
  for (int i = 0; i < p; ++i) {
    grad_grad += gg_a[i] * da[i];
  }
  for (int j = 0; j < q; ++j) {
    grad_grad += gg_b[j] * db[j];
  }

  // For second derivatives, use finite differences of the backward pass

  // Compute gradients w.r.t. a[i]
  std::vector<T> a_work(p);
  for (int i = 0; i < p; ++i) {
    for (int k = 0; k < p; ++k) {
      a_work[k] = a[k];
    }

    // Backward at a[i] + eps
    a_work[i] = a[i] + eps;
    std::vector<T> da_plus(p);
    std::vector<T> db_plus(q);
    T dz_plus;
    hypergeometric_p_f_q_backward(T(1), a_work.data(), p, b, q, z, da_plus.data(), db_plus.data(), dz_plus);

    // Backward at a[i] - eps
    a_work[i] = a[i] - eps;
    std::vector<T> da_minus(p);
    std::vector<T> db_minus(q);
    T dz_minus;
    hypergeometric_p_f_q_backward(T(1), a_work.data(), p, b, q, z, da_minus.data(), db_minus.data(), dz_minus);

    // Accumulate contributions from all gg terms
    T total = gg_z * (dz_plus - dz_minus) / (T(2) * eps);
    for (int k = 0; k < p; ++k) {
      total += gg_a[k] * (da_plus[k] - da_minus[k]) / (T(2) * eps);
    }
    for (int k = 0; k < q; ++k) {
      total += gg_b[k] * (db_plus[k] - db_minus[k]) / (T(2) * eps);
    }

    if constexpr (pfq_is_complex_v<T>) {
      out_grad_a[i] = grad * std::conj(total);
    } else {
      out_grad_a[i] = grad * total;
    }
  }

  // Compute gradients w.r.t. b[j]
  std::vector<T> b_work(q);
  for (int j = 0; j < q; ++j) {
    for (int k = 0; k < q; ++k) {
      b_work[k] = b[k];
    }

    // Backward at b[j] + eps
    b_work[j] = b[j] + eps;
    std::vector<T> da_plus(p);
    std::vector<T> db_plus(q);
    T dz_plus;
    hypergeometric_p_f_q_backward(T(1), a, p, b_work.data(), q, z, da_plus.data(), db_plus.data(), dz_plus);

    // Backward at b[j] - eps
    b_work[j] = b[j] - eps;
    std::vector<T> da_minus(p);
    std::vector<T> db_minus(q);
    T dz_minus;
    hypergeometric_p_f_q_backward(T(1), a, p, b_work.data(), q, z, da_minus.data(), db_minus.data(), dz_minus);

    // Accumulate contributions from all gg terms
    T total = gg_z * (dz_plus - dz_minus) / (T(2) * eps);
    for (int k = 0; k < p; ++k) {
      total += gg_a[k] * (da_plus[k] - da_minus[k]) / (T(2) * eps);
    }
    for (int k = 0; k < q; ++k) {
      total += gg_b[k] * (db_plus[k] - db_minus[k]) / (T(2) * eps);
    }

    if constexpr (pfq_is_complex_v<T>) {
      out_grad_b[j] = grad * std::conj(total);
    } else {
      out_grad_b[j] = grad * total;
    }
  }

  // Compute gradient w.r.t. z
  {
    // Backward at z + eps
    std::vector<T> da_plus(p);
    std::vector<T> db_plus(q);
    T dz_plus;
    hypergeometric_p_f_q_backward(T(1), a, p, b, q, z + eps, da_plus.data(), db_plus.data(), dz_plus);

    // Backward at z - eps
    std::vector<T> da_minus(p);
    std::vector<T> db_minus(q);
    T dz_minus;
    hypergeometric_p_f_q_backward(T(1), a, p, b, q, z - eps, da_minus.data(), db_minus.data(), dz_minus);

    // Accumulate contributions from all gg terms
    T total = gg_z * (dz_plus - dz_minus) / (T(2) * eps);
    for (int k = 0; k < p; ++k) {
      total += gg_a[k] * (da_plus[k] - da_minus[k]) / (T(2) * eps);
    }
    for (int k = 0; k < q; ++k) {
      total += gg_b[k] * (db_plus[k] - db_minus[k]) / (T(2) * eps);
    }

    if constexpr (pfq_is_complex_v<T>) {
      out_grad_z = grad * std::conj(total);
    } else {
      out_grad_z = grad * total;
    }
  }
}

} // namespace torchscience::kernel::special_functions
