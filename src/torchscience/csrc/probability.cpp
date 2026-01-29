// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/probability/normal.h"
#include "cpu/probability/chi2.h"
#include "cpu/probability/f.h"
#include "cpu/probability/beta.h"
#include "cpu/probability/gamma.h"
#include "cpu/probability/binomial.h"
#include "cpu/probability/poisson.h"

// Meta backend
#include "meta/probability/normal.h"
#include "meta/probability/chi2.h"
#include "meta/probability/f.h"
#include "meta/probability/beta.h"
#include "meta/probability/gamma.h"
#include "meta/probability/binomial.h"
#include "meta/probability/poisson.h"

// Autograd backend
#include "autograd/probability/normal.h"
#include "autograd/probability/chi2.h"
#include "autograd/probability/f.h"
#include "autograd/probability/beta.h"
#include "autograd/probability/gamma.h"
#include "autograd/probability/binomial.h"
#include "autograd/probability/poisson.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Normal distribution
  m.def("normal_probability_density(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  m.def("normal_probability_density_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");

  m.def("normal_log_probability_density(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  m.def("normal_log_probability_density_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");

  // Chi-squared distribution
  m.def("chi2_probability_density(Tensor x, Tensor df) -> Tensor");
  m.def("chi2_probability_density_backward(Tensor grad, Tensor x, Tensor df) -> (Tensor, Tensor)");

  // F-distribution
  m.def("f_probability_density(Tensor x, Tensor dfn, Tensor dfd) -> Tensor");
  m.def("f_probability_density_backward(Tensor grad, Tensor x, Tensor dfn, Tensor dfd) -> (Tensor, Tensor, Tensor)");

  // Beta distribution
  m.def("beta_probability_density(Tensor x, Tensor a, Tensor b) -> Tensor");
  m.def("beta_probability_density_backward(Tensor grad, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Gamma distribution
  m.def("gamma_probability_density(Tensor x, Tensor shape, Tensor scale) -> Tensor");
  m.def("gamma_probability_density_backward(Tensor grad, Tensor x, Tensor shape, Tensor scale) -> (Tensor, Tensor, Tensor)");

  // Binomial distribution
  m.def("binomial_probability_mass(Tensor k, Tensor n, Tensor p) -> Tensor");
  m.def("binomial_probability_mass_backward(Tensor grad, Tensor k, Tensor n, Tensor p) -> (Tensor, Tensor, Tensor)");

  // Poisson distribution
  m.def("poisson_probability_mass(Tensor k, Tensor rate) -> Tensor");
  m.def("poisson_probability_mass_backward(Tensor grad, Tensor k, Tensor rate) -> (Tensor, Tensor)");
}
