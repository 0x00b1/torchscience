// src/torchscience/csrc/cpu/ordinary_differential_equation/initial_value_problem/backward_euler.h
#pragma once

#include <ATen/ATen.h>
#include <functional>
#include <tuple>

namespace torchscience {
namespace cpu {
namespace ordinary_differential_equation {
namespace initial_value_problem {

template <typename scalar_t>
struct NewtonSolver {
  /**
   * Solve f(x) = 0 using Newton's method.
   *
   * @param f Function to find root of
   * @param jacobian Function returning Jacobian of f
   * @param x0 Initial guess
   * @param tol Convergence tolerance
   * @param max_iter Maximum iterations
   * @return (solution, converged) tuple
   */
  static std::tuple<at::Tensor, bool> solve(
      const std::function<at::Tensor(const at::Tensor&)>& f,
      const std::function<at::Tensor(const at::Tensor&)>& jacobian,
      const at::Tensor& x0,
      scalar_t tol,
      int max_iter
  ) {
    at::Tensor x = x0.clone();

    for (int i = 0; i < max_iter; ++i) {
      at::Tensor residual = f(x);
      scalar_t norm = at::linalg_norm(residual).item<scalar_t>();

      if (norm < tol) {
        return std::make_tuple(x, true);
      }

      at::Tensor J = jacobian(x);
      at::Tensor dx = at::linalg_solve(J, -residual.unsqueeze(-1)).squeeze(-1);
      x = x + dx;
    }

    // Check final convergence
    at::Tensor residual = f(x);
    scalar_t norm = at::linalg_norm(residual).item<scalar_t>();
    return std::make_tuple(x, norm < tol);
  }
};

template <typename scalar_t>
struct BackwardEulerStep {
  /**
   * Single step of backward Euler method.
   *
   * Solves: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
   *
   * @param f Dynamics function
   * @param jacobian_f Jacobian of dynamics w.r.t. y
   * @param t Current time
   * @param y Current state
   * @param h Step size
   * @param tol Newton tolerance
   * @param max_iter Maximum Newton iterations
   * @return (y_new, converged) tuple
   */
  static std::tuple<at::Tensor, bool> step(
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& f,
      const std::function<at::Tensor(scalar_t, const at::Tensor&)>& jacobian_f,
      scalar_t t,
      const at::Tensor& y,
      scalar_t h,
      scalar_t tol,
      int max_iter
  ) {
    scalar_t t_next = t + h;

    // Residual: g(y_new) = y_new - y - h * f(t_next, y_new)
    auto residual = [&](const at::Tensor& y_new) {
      return y_new - y - h * f(t_next, y_new);
    };

    // Jacobian: dg/dy_new = I - h * df/dy
    auto jacobian = [&](const at::Tensor& y_new) {
      int64_t n = y_new.size(0);
      at::Tensor I = at::eye(n, y_new.options());
      at::Tensor df_dy = jacobian_f(t_next, y_new);
      return I - h * df_dy;
    };

    // Initial guess: forward Euler
    at::Tensor y_guess = y + h * f(t, y);

    return NewtonSolver<scalar_t>::solve(residual, jacobian, y_guess, tol, max_iter);
  }
};

}  // namespace initial_value_problem
}  // namespace ordinary_differential_equation
}  // namespace cpu
}  // namespace torchscience
