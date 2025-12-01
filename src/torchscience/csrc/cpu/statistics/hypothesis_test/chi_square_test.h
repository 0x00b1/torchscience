#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/chi_square_test.h"
#include "../../../kernel/statistics/hypothesis_test/chi_square_test_backward.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Chi-square goodness-of-fit test - CPU implementation.
 *
 * @param observed Observed frequencies (last dimension = categories)
 * @param expected Optional expected frequencies (same shape as observed)
 * @param ddof Delta degrees of freedom
 * @return Tuple of (chi-square statistic, p-value) tensors
 */
inline std::tuple<at::Tensor, at::Tensor> chi_square_test(
    const at::Tensor& observed,
    const c10::optional<at::Tensor>& expected,
    int64_t ddof
) {
    TORCH_CHECK(
        observed.dim() >= 1,
        "chi_square_test: observed must have at least 1 dimension, got ",
        observed.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(observed.scalar_type()),
        "chi_square_test: observed must be floating-point, got ",
        observed.scalar_type()
    );

    at::Tensor observed_contig = observed.contiguous();
    int64_t k = observed_contig.size(-1);
    int64_t batch_size = observed_contig.numel() / k;

    at::Tensor expected_contig;
    if (expected.has_value()) {
        TORCH_CHECK(
            expected->sizes() == observed.sizes(),
            "chi_square_test: expected must have same shape as observed"
        );
        expected_contig = expected->contiguous();
    }

    auto output_shape = observed_contig.sizes().vec();
    output_shape.pop_back();

    auto options = observed_contig.options();
    at::Tensor statistic = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);
    at::Tensor pvalue = output_shape.empty()
        ? at::empty({}, options)
        : at::empty(output_shape, options);

    AT_DISPATCH_FLOATING_TYPES(
        observed_contig.scalar_type(),
        "chi_square_test_cpu",
        [&] {
            const scalar_t* obs_ptr = observed_contig.data_ptr<scalar_t>();
            const scalar_t* exp_ptr = expected.has_value()
                ? expected_contig.data_ptr<scalar_t>()
                : nullptr;
            scalar_t* stat_ptr = statistic.data_ptr<scalar_t>();
            scalar_t* pval_ptr = pvalue.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    const scalar_t* exp_batch = exp_ptr
                        ? exp_ptr + b * k
                        : nullptr;
                    auto [s, p] = kernel::chi_square_test<scalar_t>(
                        obs_ptr + b * k,
                        exp_batch,
                        k,
                        ddof
                    );
                    stat_ptr[b] = s;
                    pval_ptr[b] = p;
                }
            });
        }
    );

    return std::make_tuple(statistic, pvalue);
}

/**
 * Backward pass for chi-square test - CPU implementation.
 */
inline at::Tensor chi_square_test_backward(
    const at::Tensor& grad_statistic,
    const at::Tensor& observed,
    const c10::optional<at::Tensor>& expected
) {
    TORCH_CHECK(
        observed.dim() >= 1,
        "chi_square_test_backward: observed must have at least 1 dimension"
    );

    at::Tensor observed_contig = observed.contiguous();
    at::Tensor grad_contig = grad_statistic.contiguous();

    int64_t k = observed_contig.size(-1);
    int64_t batch_size = observed_contig.numel() / k;

    at::Tensor expected_contig;
    if (expected.has_value()) {
        expected_contig = expected->contiguous();
    }

    at::Tensor grad_observed = at::empty_like(observed_contig);

    AT_DISPATCH_FLOATING_TYPES(
        observed_contig.scalar_type(),
        "chi_square_test_backward_cpu",
        [&] {
            const scalar_t* obs_ptr = observed_contig.data_ptr<scalar_t>();
            const scalar_t* exp_ptr = expected.has_value()
                ? expected_contig.data_ptr<scalar_t>()
                : nullptr;
            const scalar_t* grad_stat_ptr = grad_contig.data_ptr<scalar_t>();
            scalar_t* grad_obs_ptr = grad_observed.data_ptr<scalar_t>();

            at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    const scalar_t* exp_batch = exp_ptr
                        ? exp_ptr + b * k
                        : nullptr;
                    kernel::chi_square_test_backward<scalar_t>(
                        grad_stat_ptr[b],
                        obs_ptr + b * k,
                        exp_batch,
                        grad_obs_ptr + b * k,
                        k
                    );
                }
            });
        }
    );

    return grad_observed;
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("chi_square_test", &torchscience::cpu::statistics::hypothesis_test::chi_square_test);
    m.impl("chi_square_test_backward", &torchscience::cpu::statistics::hypothesis_test::chi_square_test_backward);
}
