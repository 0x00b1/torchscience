#pragma once

#include <torch/extension.h>

namespace torchscience::autograd::probability {

class BinomialCdfFunction : public torch::autograd::Function<BinomialCdfFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& k,
      const at::Tensor& n,
      const at::Tensor& p
  ) {
    ctx->save_for_backward({k, n, p});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::binomial_cumulative_distribution", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(k, n, p);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor k = saved[0];
    at::Tensor n = saved[1];
    at::Tensor p = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::binomial_cumulative_distribution_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, k, n, p);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor binomial_cumulative_distribution(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p
) {
  return BinomialCdfFunction::apply(k, n, p);
}

class BinomialProbabilityMassFunction : public torch::autograd::Function<BinomialProbabilityMassFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& k,
      const at::Tensor& n,
      const at::Tensor& p
  ) {
    ctx->save_for_backward({k, n, p});

    at::AutoDispatchBelowAutograd guard;

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::binomial_probability_mass", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(k, n, p);
  }

  static std::vector<at::Tensor> backward(
      torch::autograd::AutogradContext* ctx,
      const std::vector<at::Tensor>& grad_outputs
  ) {
    const torch::autograd::variable_list saved = ctx->get_saved_variables();
    at::Tensor k = saved[0];
    at::Tensor n = saved[1];
    at::Tensor p = saved[2];

    at::Tensor grad_output = grad_outputs[0];

    if (!grad_output.defined()) {
      return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::AutoDispatchBelowAutograd guard;

    auto result = c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::binomial_probability_mass_backward", "")
        .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor>(
            const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(grad_output, k, n, p);

    return {std::get<0>(result), std::get<1>(result), std::get<2>(result)};
  }
};

inline at::Tensor binomial_probability_mass(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p
) {
  return BinomialProbabilityMassFunction::apply(k, n, p);
}

}  // namespace torchscience::autograd::probability

TORCH_LIBRARY_IMPL(torchscience, Autograd, m) {
  m.impl("binomial_cumulative_distribution", &torchscience::autograd::probability::binomial_cumulative_distribution);
  m.impl("binomial_probability_mass", &torchscience::autograd::probability::binomial_probability_mass);
}
