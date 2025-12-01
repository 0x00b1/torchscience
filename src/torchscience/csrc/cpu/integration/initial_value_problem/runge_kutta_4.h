// src/torchscience/csrc/cpu/integration/initial_value_problem/runge_kutta_4.h
#pragma once

#include <ATen/ATen.h>
#include <functional>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

template <typename scalar_t>
struct RungeKutta4Step {
  // Single step of classic 4th-order Runge-Kutta
  static at::Tensor step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h
  ) {
    at::Tensor k1 = f(t, y);
    at::Tensor k2 = f(t + h / 2, y + h / 2 * k1);
    at::Tensor k3 = f(t + h / 2, y + h / 2 * k2);
    at::Tensor k4 = f(t + h, y + h * k3);
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
  }
};

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
