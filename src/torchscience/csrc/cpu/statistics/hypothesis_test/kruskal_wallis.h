// src/torchscience/csrc/cpu/statistics/hypothesis_test/kruskal_wallis.h
#pragma once

#include <tuple>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "../../../kernel/statistics/hypothesis_test/kruskal_wallis.h"

namespace torchscience::cpu::statistics::hypothesis_test {

namespace kernel = torchscience::kernel::statistics::hypothesis_test;

/**
 * Kruskal-Wallis H test - CPU implementation.
 *
 * Non-parametric test for comparing distributions of k independent samples.
 *
 * @param data Concatenated data from all groups (1D tensor)
 * @param group_sizes Sizes of each group (1D tensor of integers)
 * @return Tuple of (H-statistic, p-value) tensors (scalars)
 */
inline std::tuple<at::Tensor, at::Tensor> kruskal_wallis(
    const at::Tensor& data,
    const at::Tensor& group_sizes
) {
    TORCH_CHECK(
        data.dim() == 1,
        "kruskal_wallis: data must be 1-dimensional, got ", data.dim(), "D"
    );
    TORCH_CHECK(
        group_sizes.dim() == 1,
        "kruskal_wallis: group_sizes must be 1-dimensional, got ", group_sizes.dim(), "D"
    );
    TORCH_CHECK(
        at::isFloatingType(data.scalar_type()),
        "kruskal_wallis: data must be floating-point, got ", data.scalar_type()
    );
    TORCH_CHECK(
        at::isIntegralType(group_sizes.scalar_type(), /*include_bool=*/false),
        "kruskal_wallis: group_sizes must be integer type"
    );
    TORCH_CHECK(
        !data.requires_grad(),
        "kruskal_wallis does not support gradients (rank-based test). "
        "Use with torch.no_grad() context."
    );

    at::Tensor data_contig = data.contiguous();
    at::Tensor group_sizes_contig = group_sizes.to(at::kLong).contiguous();

    int64_t n = data_contig.size(0);
    int64_t k = group_sizes_contig.size(0);

    // Create output tensors (scalars)
    auto options = data_contig.options();
    at::Tensor statistic = at::empty({}, options);
    at::Tensor pvalue = at::empty({}, options);

    AT_DISPATCH_FLOATING_TYPES(
        data_contig.scalar_type(),
        "kruskal_wallis_cpu",
        [&] {
            const scalar_t* data_ptr = data_contig.data_ptr<scalar_t>();
            const int64_t* group_sizes_ptr = group_sizes_contig.data_ptr<int64_t>();

            auto [H, p] = kernel::kruskal_wallis<scalar_t>(
                data_ptr, n, group_sizes_ptr, k
            );

            statistic.fill_(H);
            pvalue.fill_(p);
        }
    );

    return std::make_tuple(statistic, pvalue);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
    m.impl(
        "kruskal_wallis",
        &torchscience::cpu::statistics::hypothesis_test::kruskal_wallis
    );
}
