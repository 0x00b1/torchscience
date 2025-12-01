// src/torchscience/csrc/cpu/integration/initial_value_problem/dormand_prince_5.h
#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <tuple>
#include <vector>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

// Dormand-Prince 5(4) Butcher tableau coefficients
template <typename scalar_t>
struct DormandPrince5Tableau {
  static constexpr scalar_t c2 = 1.0 / 5.0;
  static constexpr scalar_t c3 = 3.0 / 10.0;
  static constexpr scalar_t c4 = 4.0 / 5.0;
  static constexpr scalar_t c5 = 8.0 / 9.0;
  static constexpr scalar_t c6 = 1.0;
  static constexpr scalar_t c7 = 1.0;

  static constexpr scalar_t a21 = 1.0 / 5.0;
  static constexpr scalar_t a31 = 3.0 / 40.0;
  static constexpr scalar_t a32 = 9.0 / 40.0;
  static constexpr scalar_t a41 = 44.0 / 45.0;
  static constexpr scalar_t a42 = -56.0 / 15.0;
  static constexpr scalar_t a43 = 32.0 / 9.0;
  static constexpr scalar_t a51 = 19372.0 / 6561.0;
  static constexpr scalar_t a52 = -25360.0 / 2187.0;
  static constexpr scalar_t a53 = 64448.0 / 6561.0;
  static constexpr scalar_t a54 = -212.0 / 729.0;
  static constexpr scalar_t a61 = 9017.0 / 3168.0;
  static constexpr scalar_t a62 = -355.0 / 33.0;
  static constexpr scalar_t a63 = 46732.0 / 5247.0;
  static constexpr scalar_t a64 = 49.0 / 176.0;
  static constexpr scalar_t a65 = -5103.0 / 18656.0;
  static constexpr scalar_t a71 = 35.0 / 384.0;
  static constexpr scalar_t a72 = 0.0;
  static constexpr scalar_t a73 = 500.0 / 1113.0;
  static constexpr scalar_t a74 = 125.0 / 192.0;
  static constexpr scalar_t a75 = -2187.0 / 6784.0;
  static constexpr scalar_t a76 = 11.0 / 84.0;

  // 5th order weights (same as a7*)
  static constexpr scalar_t b51 = 35.0 / 384.0;
  static constexpr scalar_t b52 = 0.0;
  static constexpr scalar_t b53 = 500.0 / 1113.0;
  static constexpr scalar_t b54 = 125.0 / 192.0;
  static constexpr scalar_t b55 = -2187.0 / 6784.0;
  static constexpr scalar_t b56 = 11.0 / 84.0;
  static constexpr scalar_t b57 = 0.0;

  // 4th order weights (for error estimation)
  static constexpr scalar_t b41 = 5179.0 / 57600.0;
  static constexpr scalar_t b42 = 0.0;
  static constexpr scalar_t b43 = 7571.0 / 16695.0;
  static constexpr scalar_t b44 = 393.0 / 640.0;
  static constexpr scalar_t b45 = -92097.0 / 339200.0;
  static constexpr scalar_t b46 = 187.0 / 2100.0;
  static constexpr scalar_t b47 = 1.0 / 40.0;
};

template <typename scalar_t>
struct DormandPrince5Step {
  // Single step of Dormand-Prince 5(4) method
  // Returns: (y_new, error_estimate, k7) where k7 can be reused as k1 for next step (FSAL)
  static std::tuple<at::Tensor, at::Tensor, at::Tensor> step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h,
      const at::Tensor& k1  // Can reuse from previous step (FSAL)
  ) {
    using T = DormandPrince5Tableau<scalar_t>;

    at::Tensor k2 = f(t + T::c2 * h, y + h * T::a21 * k1);
    at::Tensor k3 = f(t + T::c3 * h, y + h * (T::a31 * k1 + T::a32 * k2));
    at::Tensor k4 = f(t + T::c4 * h, y + h * (T::a41 * k1 + T::a42 * k2 + T::a43 * k3));
    at::Tensor k5 = f(t + T::c5 * h, y + h * (T::a51 * k1 + T::a52 * k2 + T::a53 * k3 + T::a54 * k4));
    at::Tensor k6 = f(t + T::c6 * h, y + h * (T::a61 * k1 + T::a62 * k2 + T::a63 * k3 + T::a64 * k4 + T::a65 * k5));

    // 5th order solution
    at::Tensor y_new = y + h * (T::b51 * k1 + T::b52 * k2 + T::b53 * k3 + T::b54 * k4 + T::b55 * k5 + T::b56 * k6);

    // k7 for FSAL property (equals f(t+h, y_new))
    at::Tensor k7 = f(t + h, y_new);

    // Error estimate (difference between 5th and 4th order solutions)
    at::Tensor error = h * (
        (T::b51 - T::b41) * k1 +
        (T::b52 - T::b42) * k2 +
        (T::b53 - T::b43) * k3 +
        (T::b54 - T::b44) * k4 +
        (T::b55 - T::b45) * k5 +
        (T::b56 - T::b46) * k6 +
        (T::b57 - T::b47) * k7
    );

    return std::make_tuple(y_new, error, k7);
  }
};

// Error norm computation for step size control
template <typename scalar_t>
scalar_t compute_error_norm(
    const at::Tensor& error,
    const at::Tensor& y,
    const at::Tensor& y_new,
    scalar_t atol,
    scalar_t rtol
) {
  at::Tensor scale = atol + rtol * at::maximum(at::abs(y), at::abs(y_new));
  at::Tensor normalized_error = error / scale;
  scalar_t err_norm = std::sqrt(at::mean(normalized_error * normalized_error).item<scalar_t>());
  return err_norm;
}

// Step size adjustment
template <typename scalar_t>
scalar_t adjust_step_size(
    scalar_t dt,
    scalar_t err_norm,
    scalar_t dt_min,
    scalar_t dt_max,
    scalar_t safety = 0.9,
    scalar_t min_factor = 0.1,
    scalar_t max_factor = 5.0
) {
  scalar_t factor;
  if (err_norm == 0) {
    factor = max_factor;
  } else {
    // PI controller: (1/err)^(1/5) for 5th order method
    factor = safety * std::pow(1.0 / err_norm, 0.2);
  }
  factor = std::max(min_factor, std::min(factor, max_factor));

  scalar_t dt_new = dt * factor;
  if (dt_max > 0) {
    dt_new = std::min(dt_new, dt_max);
  }
  if (dt_min > 0) {
    dt_new = std::max(dt_new, dt_min);
  }
  return dt_new;
}

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
