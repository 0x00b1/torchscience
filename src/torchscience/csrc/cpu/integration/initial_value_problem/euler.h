// src/torchscience/csrc/cpu/integration/initial_value_problem/euler.h
#pragma once

#include <ATen/ATen.h>
#include <functional>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

template <typename scalar_t>
struct EulerStep {
  // Single step of forward Euler method
  static at::Tensor step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h
  ) {
    at::Tensor k = f(t, y);
    return y + h * k;
  }
};

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
