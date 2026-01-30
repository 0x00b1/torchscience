#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/binomial_cumulative_distribution.h"
#include "../../kernel/probability/binomial_cumulative_distribution_backward.h"
#include "../../kernel/probability/binomial_log_probability_mass.h"
#include "../../kernel/probability/binomial_log_probability_mass_backward.h"
#include "../../kernel/probability/binomial_probability_mass.h"
#include "../../kernel/probability/binomial_probability_mass_backward.h"
#include "../../kernel/probability/binomial_survival.h"
#include "../../kernel/probability/binomial_survival_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available

at::Tensor binomial_cumulative_distribution(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({k, n, p});
  auto k_b = tensors[0].contiguous();
  auto n_b = tensors[1].contiguous();
  auto p_b = tensors[2].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_cumulative_distribution_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::binomial_cumulative_distribution<scalar_t>(
                k_data[i], n_data[i], p_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({grad, k, n, p});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto n_b = tensors[2].contiguous();
  auto p_b = tensors[3].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_n = at::empty_like(n_b);
  auto grad_p = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_cumulative_distribution_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_n_data = grad_n.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gn, gp] = kernel::probability::binomial_cumulative_distribution_backward<scalar_t>(
                grad_data[i], k_data[i], n_data[i], p_data[i]);
            grad_k_data[i] = gk;
            grad_n_data[i] = gn;
            grad_p_data[i] = gp;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_n, n),
      reduce_grad(grad_p, p));
}

at::Tensor binomial_probability_mass(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({k, n, p});
  auto k_b = tensors[0].contiguous();
  auto n_b = tensors[1].contiguous();
  auto p_b = tensors[2].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_probability_mass_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::binomial_probability_mass<scalar_t>(
                k_data[i], n_data[i], p_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_probability_mass_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({grad, k, n, p});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto n_b = tensors[2].contiguous();
  auto p_b = tensors[3].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_n = at::empty_like(n_b);
  auto grad_p = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_probability_mass_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_n_data = grad_n.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gn, gp] = kernel::probability::binomial_probability_mass_backward<scalar_t>(
                grad_data[i], k_data[i], n_data[i], p_data[i]);
            grad_k_data[i] = gk;
            grad_n_data[i] = gn;
            grad_p_data[i] = gp;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_n, n),
      reduce_grad(grad_p, p));
}

at::Tensor binomial_log_probability_mass(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({k, n, p});
  auto k_b = tensors[0].contiguous();
  auto n_b = tensors[1].contiguous();
  auto p_b = tensors[2].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_log_probability_mass_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::binomial_log_probability_mass<scalar_t>(
                k_data[i], n_data[i], p_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_log_probability_mass_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({grad, k, n, p});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto n_b = tensors[2].contiguous();
  auto p_b = tensors[3].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_n = at::empty_like(n_b);
  auto grad_p = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_log_probability_mass_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_n_data = grad_n.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gn, gp] = kernel::probability::binomial_log_probability_mass_backward<scalar_t>(
                grad_data[i], k_data[i], n_data[i], p_data[i]);
            grad_k_data[i] = gk;
            grad_n_data[i] = gn;
            grad_p_data[i] = gp;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_n, n),
      reduce_grad(grad_p, p));
}

at::Tensor binomial_survival(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({k, n, p});
  auto k_b = tensors[0].contiguous();
  auto n_b = tensors[1].contiguous();
  auto p_b = tensors[2].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_survival_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::binomial_survival<scalar_t>(
                k_data[i], n_data[i], p_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto tensors = at::broadcast_tensors({grad, k, n, p});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto n_b = tensors[2].contiguous();
  auto p_b = tensors[3].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_n = at::empty_like(n_b);
  auto grad_p = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "binomial_survival_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto n_data = n_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_n_data = grad_n.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gn, gp] = kernel::probability::binomial_survival_backward<scalar_t>(
                grad_data[i], k_data[i], n_data[i], p_data[i]);
            grad_k_data[i] = gk;
            grad_n_data[i] = gn;
            grad_p_data[i] = gp;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_n, n),
      reduce_grad(grad_p, p));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("binomial_cumulative_distribution", &binomial_cumulative_distribution);
  m.impl("binomial_cumulative_distribution_backward", &binomial_cumulative_distribution_backward);
  m.impl("binomial_log_probability_mass", &binomial_log_probability_mass);
  m.impl("binomial_log_probability_mass_backward", &binomial_log_probability_mass_backward);
  m.impl("binomial_probability_mass", &binomial_probability_mass);
  m.impl("binomial_probability_mass_backward", &binomial_probability_mass_backward);
  m.impl("binomial_survival", &binomial_survival);
  m.impl("binomial_survival_backward", &binomial_survival_backward);
}

}  // namespace torchscience::cpu::probability
