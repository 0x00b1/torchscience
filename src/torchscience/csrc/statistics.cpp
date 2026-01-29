// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// Descriptive statistics - CPU
#include "cpu/statistics/descriptive/kurtosis.h"
#include "cpu/statistics/descriptive/histogram.h"

// Hypothesis tests - CPU
#include "cpu/statistics/hypothesis_test/one_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/two_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/paired_t_test.h"
#include "cpu/statistics/hypothesis_test/shapiro_wilk.h"
#include "cpu/statistics/hypothesis_test/anderson_darling.h"
#include "cpu/statistics/hypothesis_test/f_oneway.h"
#include "cpu/statistics/hypothesis_test/jarque_bera.h"
#include "cpu/statistics/hypothesis_test/chi_square_test.h"
#include "cpu/statistics/hypothesis_test/mann_whitney_u.h"
#include "cpu/statistics/hypothesis_test/wilcoxon_signed_rank.h"
#include "cpu/statistics/hypothesis_test/kruskal_wallis.h"

// Meta backend
#include "meta/statistics/descriptive/kurtosis.h"
#include "meta/statistics/descriptive/histogram.h"
#include "meta/statistics/hypothesis_test/one_sample_t_test.h"
#include "meta/statistics/hypothesis_test/two_sample_t_test.h"
#include "meta/statistics/hypothesis_test/paired_t_test.h"
#include "meta/statistics/hypothesis_test/shapiro_wilk.h"
#include "meta/statistics/hypothesis_test/anderson_darling.h"
#include "meta/statistics/hypothesis_test/f_oneway.h"
#include "meta/statistics/hypothesis_test/jarque_bera.h"
#include "meta/statistics/hypothesis_test/chi_square_test.h"
#include "meta/statistics/hypothesis_test/mann_whitney_u.h"
#include "meta/statistics/hypothesis_test/wilcoxon_signed_rank.h"
#include "meta/statistics/hypothesis_test/kruskal_wallis.h"

// Autograd backend
#include "autograd/statistics/descriptive/kurtosis.h"
#include "autograd/statistics/hypothesis_test/jarque_bera.h"
#include "autograd/statistics/hypothesis_test/f_oneway.h"
#include "autograd/statistics/hypothesis_test/chi_square_test.h"

// Autocast backend
#include "autocast/statistics/descriptive/kurtosis.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/statistics/descriptive/kurtosis.cu"
#include "cuda/statistics/descriptive/histogram.cu"
#endif

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Descriptive statistics
  m.def("kurtosis(Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  m.def("kurtosis_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  m.def("kurtosis_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> (Tensor, Tensor)");

  m.def("histogram(Tensor input, int bins, float[]? range, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");
  m.def("histogram_edges(Tensor input, Tensor bins, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");

  // Parametric hypothesis tests
  m.def("one_sample_t_test(Tensor input, float popmean, str alternative) -> (Tensor, Tensor, Tensor)");
  m.def("two_sample_t_test(Tensor input1, Tensor input2, bool equal_var, str alternative) -> (Tensor, Tensor, Tensor)");
  m.def("paired_t_test(Tensor input1, Tensor input2, str alternative) -> (Tensor, Tensor, Tensor)");

  // Normality tests
  m.def("shapiro_wilk(Tensor input) -> (Tensor, Tensor)");
  m.def("anderson_darling(Tensor input) -> (Tensor, Tensor, Tensor)");
  m.def("jarque_bera(Tensor input) -> (Tensor, Tensor)");
  m.def("jarque_bera_backward(Tensor grad_statistic, Tensor input) -> Tensor");

  // ANOVA
  m.def("f_oneway(Tensor data, Tensor group_sizes) -> (Tensor, Tensor)");
  m.def("f_oneway_backward(Tensor grad_statistic, Tensor data, Tensor group_sizes) -> Tensor");

  // Chi-square test
  m.def("chi_square_test(Tensor observed, Tensor? expected, int ddof) -> (Tensor, Tensor)");
  m.def("chi_square_test_backward(Tensor grad_statistic, Tensor observed, Tensor? expected) -> Tensor");

  // Non-parametric rank-based tests (no gradients)
  m.def("mann_whitney_u(Tensor x, Tensor y, str alternative) -> (Tensor, Tensor)");
  m.def("wilcoxon_signed_rank(Tensor x, Tensor? y, str alternative, str zero_method) -> (Tensor, Tensor)");
  m.def("kruskal_wallis(Tensor data, Tensor group_sizes) -> (Tensor, Tensor)");
}
