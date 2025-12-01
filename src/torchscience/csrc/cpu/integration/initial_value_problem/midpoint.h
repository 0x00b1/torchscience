// src/torchscience/csrc/cpu/integration/initial_value_problem/midpoint.h
#pragma once

#include <ATen/ATen.h>
#include <functional>

namespace torchscience {
namespace cpu {
namespace integration {
namespace initial_value_problem {

template <typename scalar_t>
struct MidpointStep {
  // Single step of explicit midpoint method (RK2)
  static at::Tensor step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h
  ) {
    at::Tensor k1 = f(t, y);
    at::Tensor k2 = f(t + h / 2, y + h / 2 * k1);
    return y + h * k2;
  }
};

}  // namespace initial_value_problem
}  // namespace integration
}  // namespace cpu
}  // namespace torchscience
