#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/poisson_cumulative_distribution.h"
#include "../../kernel/probability/poisson_cumulative_distribution_backward.h"
#include "../../kernel/probability/poisson_log_probability_mass.h"
#include "../../kernel/probability/poisson_log_probability_mass_backward.h"
#include "../../kernel/probability/poisson_probability_mass.h"
#include "../../kernel/probability/poisson_probability_mass_backward.h"
#include "../../kernel/probability/poisson_survival.h"
#include "../../kernel/probability/poisson_survival_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available

at::Tensor poisson_cumulative_distribution(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({k, rate});
  auto k_b = tensors[0].contiguous();
  auto rate_b = tensors[1].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_cumulative_distribution_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::poisson_cumulative_distribution<scalar_t>(
                k_data[i], rate_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor> poisson_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({grad, k, rate});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto rate_b = tensors[2].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_rate = at::empty_like(rate_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_cumulative_distribution_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_rate_data = grad_rate.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gr] = kernel::probability::poisson_cumulative_distribution_backward<scalar_t>(
                grad_data[i], k_data[i], rate_data[i]);
            grad_k_data[i] = gk;
            grad_rate_data[i] = gr;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_rate, rate));
}

at::Tensor poisson_probability_mass(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({k, rate});
  auto k_b = tensors[0].contiguous();
  auto rate_b = tensors[1].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_probability_mass_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::poisson_probability_mass<scalar_t>(
                k_data[i], rate_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor> poisson_probability_mass_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({grad, k, rate});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto rate_b = tensors[2].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_rate = at::empty_like(rate_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_probability_mass_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_rate_data = grad_rate.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gr] = kernel::probability::poisson_probability_mass_backward<scalar_t>(
                grad_data[i], k_data[i], rate_data[i]);
            grad_k_data[i] = gk;
            grad_rate_data[i] = gr;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_rate, rate));
}

at::Tensor poisson_log_probability_mass(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({k, rate});
  auto k_b = tensors[0].contiguous();
  auto rate_b = tensors[1].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_log_probability_mass_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::poisson_log_probability_mass<scalar_t>(
                k_data[i], rate_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor> poisson_log_probability_mass_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({grad, k, rate});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto rate_b = tensors[2].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_rate = at::empty_like(rate_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_log_probability_mass_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_rate_data = grad_rate.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gr] = kernel::probability::poisson_log_probability_mass_backward<scalar_t>(
                grad_data[i], k_data[i], rate_data[i]);
            grad_k_data[i] = gk;
            grad_rate_data[i] = gr;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_rate, rate));
}

at::Tensor poisson_survival(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({k, rate});
  auto k_b = tensors[0].contiguous();
  auto rate_b = tensors[1].contiguous();

  auto output = at::empty_like(k_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_survival_cpu", [&] {
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t numel = output.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::poisson_survival<scalar_t>(
                k_data[i], rate_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor> poisson_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto tensors = at::broadcast_tensors({grad, k, rate});
  auto grad_b = tensors[0].contiguous();
  auto k_b = tensors[1].contiguous();
  auto rate_b = tensors[2].contiguous();

  auto grad_k = at::empty_like(k_b);
  auto grad_rate = at::empty_like(rate_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, k.scalar_type(), "poisson_survival_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto k_data = k_b.data_ptr<scalar_t>();
        auto rate_data = rate_b.data_ptr<scalar_t>();
        auto grad_k_data = grad_k.data_ptr<scalar_t>();
        auto grad_rate_data = grad_rate.data_ptr<scalar_t>();
        int64_t numel = grad_b.numel();

        at::parallel_for(0, numel, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gk, gr] = kernel::probability::poisson_survival_backward<scalar_t>(
                grad_data[i], k_data[i], rate_data[i]);
            grad_k_data[i] = gk;
            grad_rate_data[i] = gr;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_k, k),
      reduce_grad(grad_rate, rate));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("poisson_cumulative_distribution", &poisson_cumulative_distribution);
  m.impl("poisson_cumulative_distribution_backward", &poisson_cumulative_distribution_backward);
  m.impl("poisson_log_probability_mass", &poisson_log_probability_mass);
  m.impl("poisson_log_probability_mass_backward", &poisson_log_probability_mass_backward);
  m.impl("poisson_probability_mass", &poisson_probability_mass);
  m.impl("poisson_probability_mass_backward", &poisson_probability_mass_backward);
  m.impl("poisson_survival", &poisson_survival);
  m.impl("poisson_survival_backward", &poisson_survival_backward);
}

}  // namespace torchscience::cpu::probability
