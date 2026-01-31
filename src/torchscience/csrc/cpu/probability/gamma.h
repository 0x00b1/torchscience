#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/probability/gamma_cumulative_distribution.h"
#include "../../kernel/probability/gamma_cumulative_distribution_backward.h"
#include "../../kernel/probability/gamma_probability_density.h"
#include "../../kernel/probability/gamma_probability_density_backward.h"
#include "../../kernel/probability/gamma_quantile.h"
#include "../../kernel/probability/gamma_quantile_backward.h"
#include "../../kernel/probability/gamma_log_probability_density.h"
#include "../../kernel/probability/gamma_log_probability_density_backward.h"
#include "../../kernel/probability/gamma_survival.h"
#include "../../kernel/probability/gamma_survival_backward.h"

namespace torchscience::cpu::probability {

// reduce_grad is defined in normal.h and already available

at::Tensor gamma_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({x, shape, scale});
  auto x_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_cumulative_distribution_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_cumulative_distribution<scalar_t>(
                x_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, x, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_cumulative_distribution_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gs, gsc] = kernel::probability::gamma_cumulative_distribution_backward<scalar_t>(
                grad_data[i], x_data[i], shape_data[i], scale_data[i]);
            grad_x_data[i] = gx;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

at::Tensor gamma_probability_density(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({x, shape, scale});
  auto x_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_probability_density<scalar_t>(
                x_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, x, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gs, gsc] = kernel::probability::gamma_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], shape_data[i], scale_data[i]);
            grad_x_data[i] = gx;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

at::Tensor gamma_quantile(
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({p, shape, scale});
  auto p_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(p_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "gamma_quantile_cpu", [&] {
        auto p_data = p_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_quantile<scalar_t>(
                p_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, p, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto p_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_p = at::empty_like(p_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, p.scalar_type(), "gamma_quantile_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto p_data = p_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_p_data = grad_p.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gp, gs, gsc] = kernel::probability::gamma_quantile_backward<scalar_t>(
                grad_data[i], p_data[i], shape_data[i], scale_data[i]);
            grad_p_data[i] = gp;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_p, p),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

at::Tensor gamma_log_probability_density(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({x, shape, scale});
  auto x_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_log_probability_density_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_log_probability_density<scalar_t>(
                x_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_log_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, x, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_log_probability_density_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gs, gsc] = kernel::probability::gamma_log_probability_density_backward<scalar_t>(
                grad_data[i], x_data[i], shape_data[i], scale_data[i]);
            grad_x_data[i] = gx;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

at::Tensor gamma_survival(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({x, shape, scale});
  auto x_b = tensors[0].contiguous();
  auto shape_b = tensors[1].contiguous();
  auto scale_b = tensors[2].contiguous();

  auto output = at::empty_like(x_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_survival_cpu", [&] {
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto out_data = output.data_ptr<scalar_t>();
        int64_t n = output.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            out_data[i] = kernel::probability::gamma_survival<scalar_t>(
                x_data[i], shape_data[i], scale_data[i]);
          }
        });
      });

  return output;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto tensors = at::broadcast_tensors({grad, x, shape, scale});
  auto grad_b = tensors[0].contiguous();
  auto x_b = tensors[1].contiguous();
  auto shape_b = tensors[2].contiguous();
  auto scale_b = tensors[3].contiguous();

  auto grad_x = at::empty_like(x_b);
  auto grad_shape = at::empty_like(shape_b);
  auto grad_scale = at::empty_like(scale_b);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "gamma_survival_backward_cpu", [&] {
        auto grad_data = grad_b.data_ptr<scalar_t>();
        auto x_data = x_b.data_ptr<scalar_t>();
        auto shape_data = shape_b.data_ptr<scalar_t>();
        auto scale_data = scale_b.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();
        auto grad_shape_data = grad_shape.data_ptr<scalar_t>();
        auto grad_scale_data = grad_scale.data_ptr<scalar_t>();
        int64_t n = grad_b.numel();

        at::parallel_for(0, n, 1000, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            auto [gx, gs, gsc] = kernel::probability::gamma_survival_backward<scalar_t>(
                grad_data[i], x_data[i], shape_data[i], scale_data[i]);
            grad_x_data[i] = gx;
            grad_shape_data[i] = gs;
            grad_scale_data[i] = gsc;
          }
        });
      });

  return std::make_tuple(
      reduce_grad(grad_x, x),
      reduce_grad(grad_shape, shape),
      reduce_grad(grad_scale, scale));
}

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("gamma_cumulative_distribution", &gamma_cumulative_distribution);
  m.impl("gamma_cumulative_distribution_backward", &gamma_cumulative_distribution_backward);
  m.impl("gamma_probability_density", &gamma_probability_density);
  m.impl("gamma_probability_density_backward", &gamma_probability_density_backward);
  m.impl("gamma_quantile", &gamma_quantile);
  m.impl("gamma_quantile_backward", &gamma_quantile_backward);
  m.impl("gamma_log_probability_density", &gamma_log_probability_density);
  m.impl("gamma_log_probability_density_backward", &gamma_log_probability_density_backward);
  m.impl("gamma_survival", &gamma_survival);
  m.impl("gamma_survival_backward", &gamma_survival_backward);
}

}  // namespace torchscience::cpu::probability
