// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/linear_algebra/symmetric_generalized_eigenvalue.h"
#include "cpu/linear_algebra/generalized_eigenvalue.h"
#include "cpu/linear_algebra/schur_decomposition.h"
#include "cpu/linear_algebra/polar_decomposition.h"
#include "cpu/linear_algebra/hessenberg.h"
#include "cpu/linear_algebra/generalized_schur.h"
#include "cpu/linear_algebra/jordan_decomposition.h"
#include "cpu/linear_algebra/pivoted_lu.h"
#include "cpu/linear_algebra/pivoted_qr.h"
#include "cpu/linear_algebra/rank_revealing_qr.h"
#include "cpu/linear_algebra/ldl_decomposition.h"

// Meta backend
#include "meta/linear_algebra/symmetric_generalized_eigenvalue.h"
#include "meta/linear_algebra/generalized_eigenvalue.h"
#include "meta/linear_algebra/schur_decomposition.h"
#include "meta/linear_algebra/polar_decomposition.h"
#include "meta/linear_algebra/hessenberg.h"
#include "meta/linear_algebra/generalized_schur.h"
#include "meta/linear_algebra/jordan_decomposition.h"
#include "meta/linear_algebra/pivoted_lu.h"
#include "meta/linear_algebra/pivoted_qr.h"
#include "meta/linear_algebra/rank_revealing_qr.h"
#include "meta/linear_algebra/ldl_decomposition.h"

// Autograd backend
#include "autograd/linear_algebra/symmetric_generalized_eigenvalue.h"
#include "autograd/linear_algebra/polar_decomposition.h"

// Autocast backend
#include "autocast/linear_algebra/symmetric_generalized_eigenvalue.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // Generalized eigenvalue problems
  m.def("symmetric_generalized_eigenvalue(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  m.def("generalized_eigenvalue(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // Matrix decompositions
  m.def("schur_decomposition(Tensor a, str output='real') -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("polar_decomposition(Tensor a, str side='right') -> (Tensor, Tensor, Tensor)");
  m.def("hessenberg(Tensor a) -> (Tensor, Tensor, Tensor)");
  m.def("generalized_schur(Tensor a, Tensor b, str output='real') -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("jordan_decomposition(Tensor a) -> (Tensor, Tensor, Tensor)");

  // Pivoted factorizations
  m.def("pivoted_lu(Tensor a) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("pivoted_qr(Tensor a) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("rank_revealing_qr(Tensor a, float tol=1e-10) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("ldl_decomposition(Tensor a) -> (Tensor, Tensor, Tensor, Tensor)");
}
