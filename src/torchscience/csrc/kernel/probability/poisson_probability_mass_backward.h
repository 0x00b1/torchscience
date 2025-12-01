#pragma once

#include <cmath>
#include <tuple>
#include "poisson_probability_mass.h"

namespace torchscience::kernel::probability {

// Backward for Poisson probability mass function
// PMF = rate^k * exp(-rate) / k!
// dPMF/drate = PMF * (k/rate - 1)
//
// k is discrete, so its gradient is 0
template <typename T>
std::tuple<T, T> poisson_probability_mass_backward(T gradient, T k, T rate) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0) || rate <= T(0)) {
    return {T(0), T(0)};
  }

  T pmf = poisson_probability_mass(k, rate);

  // dPMF/drate = PMF * (k/rate - 1)
  T grad_rate = gradient * pmf * (k / rate - T(1));

  return {T(0), grad_rate};
}

}  // namespace torchscience::kernel::probability
