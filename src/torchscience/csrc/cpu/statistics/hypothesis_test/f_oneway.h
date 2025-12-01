#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/f_oneway.h"
#include "../../../kernel/statistics/hypothesis_test/f_oneway_backward.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * F-oneway CPU implementation.
 *
 * @param data Concatenated data from all groups
 * @param group_sizes Tensor of group sizes
 * @return Tuple of (F-statistic, p-value) tensors
 */
inline std::tuple<at::Tensor, at::Tensor> f_oneway(
    const at::Tensor& data,
    const at::Tensor& group_sizes
) {
    TORCH_CHECK(data.dim() == 1, "f_oneway: data must be 1D");
    TORCH_CHECK(group_sizes.dim() == 1, "f_oneway: group_sizes must be 1D");
    TORCH_CHECK(
        at::isFloatingType(data.scalar_type()),
        "f_oneway: data must be floating-point"
    );

    at::Tensor data_contig = data.contiguous();
    at::Tensor sizes_contig = group_sizes.contiguous().to(at::kLong);

    int64_t k = sizes_contig.size(0);

    auto options = data_contig.options();
    at::Tensor statistic = at::empty({}, options);
    at::Tensor pvalue = at::empty({}, options);

    AT_DISPATCH_FLOATING_TYPES(data_contig.scalar_type(), "f_oneway_cpu", [&] {
        auto [s, p] = kernel::f_oneway<scalar_t>(
            data_contig.data_ptr<scalar_t>(),
            sizes_contig.data_ptr<int64_t>(),
            k
        );
        statistic.fill_(s);
        pvalue.fill_(p);
    });

    return std::make_tuple(statistic, pvalue);
}

/**
 * F-oneway backward CPU implementation.
 *
 * @param grad_statistic Gradient w.r.t. F-statistic
 * @param data Concatenated data from all groups
 * @param group_sizes Tensor of group sizes
 * @return Gradient w.r.t. data
 */
inline at::Tensor f_oneway_backward(
    const at::Tensor& grad_statistic,
    const at::Tensor& data,
    const at::Tensor& group_sizes
) {
    at::Tensor data_contig = data.contiguous();
    at::Tensor sizes_contig = group_sizes.contiguous().to(at::kLong);

    int64_t k = sizes_contig.size(0);

    at::Tensor grad_data = at::empty_like(data_contig);

    AT_DISPATCH_FLOATING_TYPES(data_contig.scalar_type(), "f_oneway_backward_cpu", [&] {
        kernel::f_oneway_backward<scalar_t>(
            grad_statistic.item<scalar_t>(),
            data_contig.data_ptr<scalar_t>(),
            sizes_contig.data_ptr<int64_t>(),
            k,
            grad_data.data_ptr<scalar_t>()
        );
    });

    return grad_data;
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl("f_oneway", &torchscience::cpu::statistics::hypothesis_test::f_oneway);
    m.impl("f_oneway_backward", &torchscience::cpu::statistics::hypothesis_test::f_oneway_backward);
}
