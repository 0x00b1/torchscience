#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/beta_cumulative_distribution.h"
#include "../../kernel/probability/beta_cumulative_distribution_backward.h"
#include "../../kernel/probability/beta_probability_density.h"
#include "../../kernel/probability/beta_probability_density_backward.h"
#include "../../kernel/probability/beta_quantile.h"
#include "../../kernel/probability/beta_quantile_backward.h"
#include "../../kernel/probability/beta_log_probability_density.h"
#include "../../kernel/probability/beta_log_probability_density_backward.h"
#include "../../kernel/probability/beta_survival.h"
#include "../../kernel/probability/beta_survival_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available

at::Tensor beta_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({x, a, b});
  auto x_b = tensors[0].contiguous();
  auto a_b = tensors[1].contiguous();
  auto b_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_cumulative_distribution_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::beta_cumulative_distribution<scalar_t>(
                x_data[i], a_data[i], b_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({grad, x, a, b});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto a_b = tensors[2].contiguous();
  auto b_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_a = at::empty_like(a_b);
  auto grad_b_out = at::empty_like(b_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_cumulative_distribution_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_a_data = grad_a.data_ptr<scalar_t>();
        auto grad_b_data = grad_b_out.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, ga, gb] = kernel::probability::beta_cumulative_distribution_backward<scalar_t>(
                grad_data[i], x_data[i], a_data[i], b_data[i]);
            grad_x_data[i] = gx;
            grad_a_data[i] = ga;
            grad_b_data[i] = gb;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_a, a),
      reduce_grad(grad_b_out, b));
}

at::Tensor beta_probability_density(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({x, a, b});
  auto x_b = tensors[0].contiguous();
  auto a_b = tensors[1].contiguous();
  auto b_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::beta_probability_density<scalar_t>(
                x_data[i], a_data[i], b_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({grad, x, a, b});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto a_b = tensors[2].contiguous();
  auto b_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_a = at::empty_like(a_b);
  auto grad_b_out = at::empty_like(b_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_a_data = grad_a.data_ptr<scalar_t>();
        auto grad_b_data = grad_b_out.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, ga, gb] = kernel::probability::beta_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], a_data[i], b_data[i]);
            grad_x_data[i] = gx;
            grad_a_data[i] = ga;
            grad_b_data[i] = gb;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_a, a),
      reduce_grad(grad_b_out, b));
}

at::Tensor beta_quantile(
    const at::Tensor& p,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({p, a, b});
  auto p_b = tensors[0].contiguous();
  auto a_b = tensors[1].contiguous();
  auto b_b = tensors[2].contiguous();

  auto output = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "beta_quantile_cpu", [&] {
        auto p_data = p_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::beta_quantile<scalar_t>(
                p_data[i], a_data[i], b_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({grad, p, a, b});
  auto grad_b = tensors[0].contiguous();
  auto p_b = tensors[1].contiguous();
  auto a_b = tensors[2].contiguous();
  auto b_b = tensors[3].contiguous();

  auto grad_p = at::empty_like(p_b);
  auto grad_a = at::empty_like(a_b);
  auto grad_b_out = at::empty_like(b_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "beta_quantile_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        auto grad_a_data = grad_a.data_ptr<scalar_t>();
        auto grad_b_data = grad_b_out.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gp, ga, gb] = kernel::probability::beta_quantile_backward<scalar_t>(
                grad_data[i], p_data[i], a_data[i], b_data[i]);
            grad_p_data[i] = gp;
            grad_a_data[i] = ga;
            grad_b_data[i] = gb;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_p, p),
      reduce_grad(grad_a, a),
      reduce_grad(grad_b_out, b));
}

at::Tensor beta_log_probability_density(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({x, a, b});
  auto x_b = tensors[0].contiguous();
  auto a_b = tensors[1].contiguous();
  auto b_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_log_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::beta_log_probability_density<scalar_t>(
                x_data[i], a_data[i], b_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_log_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({grad, x, a, b});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto a_b = tensors[2].contiguous();
  auto b_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_a = at::empty_like(a_b);
  auto grad_b_out = at::empty_like(b_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_log_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_a_data = grad_a.data_ptr<scalar_t>();
        auto grad_b_data = grad_b_out.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, ga, gb] = kernel::probability::beta_log_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], a_data[i], b_data[i]);
            grad_x_data[i] = gx;
            grad_a_data[i] = ga;
            grad_b_data[i] = gb;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_a, a),
      reduce_grad(grad_b_out, b));
}

at::Tensor beta_survival(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({x, a, b});
  auto x_b = tensors[0].contiguous();
  auto a_b = tensors[1].contiguous();
  auto b_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_survival_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::beta_survival<scalar_t>(
                x_data[i], a_data[i], b_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto tensors = at::broadcast_tensors({grad, x, a, b});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto a_b = tensors[2].contiguous();
  auto b_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_a = at::empty_like(a_b);
  auto grad_b_out = at::empty_like(b_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "beta_survival_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto a_data = a_b.data_ptr<scalar_t>();
        auto b_data = b_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_a_data = grad_a.data_ptr<scalar_t>();
        auto grad_b_data = grad_b_out.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, ga, gb] = kernel::probability::beta_survival_backward<scalar_t>(
                grad_data[i], x_data[i], a_data[i], b_data[i]);
            grad_x_data[i] = gx;
            grad_a_data[i] = ga;
            grad_b_data[i] = gb;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_a, a),
      reduce_grad(grad_b_out, b));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("beta_cumulative_distribution", &beta_cumulative_distribution);
  m.impl("beta_cumulative_distribution_backward", &beta_cumulative_distribution_backward);
  m.impl("beta_probability_density", &beta_probability_density);
  m.impl("beta_probability_density_backward", &beta_probability_density_backward);
  m.impl("beta_quantile", &beta_quantile);
  m.impl("beta_quantile_backward", &beta_quantile_backward);
  m.impl("beta_log_probability_density", &beta_log_probability_density);
  m.impl("beta_log_probability_density_backward", &beta_log_probability_density_backward);
  m.impl("beta_survival", &beta_survival);
  m.impl("beta_survival_backward", &beta_survival_backward);
}

}  // namespace torchscience::cpu::probability
