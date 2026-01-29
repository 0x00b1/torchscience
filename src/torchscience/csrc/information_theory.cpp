// Copyright (c) torchscience contributors.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <torch/library.h>

// CPU backend
#include "cpu/information_theory/kullback_leibler_divergence.h"
#include "cpu/information_theory/jensen_shannon_divergence.h"
#include "cpu/information_theory/shannon_entropy.h"
#include "cpu/information_theory/joint_entropy.h"
#include "cpu/information_theory/conditional_entropy.h"
#include "cpu/information_theory/mutual_information.h"
#include "cpu/information_theory/pointwise_mutual_information.h"
#include "cpu/information_theory/cross_entropy.h"
#include "cpu/information_theory/chi_squared_divergence.h"
#include "cpu/information_theory/renyi_entropy.h"
#include "cpu/information_theory/tsallis_entropy.h"
#include "cpu/information_theory/renyi_divergence.h"
#include "cpu/information_theory/conditional_mutual_information.h"
#include "cpu/information_theory/total_correlation.h"
#include "cpu/information_theory/dual_total_correlation.h"
#include "cpu/information_theory/interaction_information.h"
#include "cpu/information_theory/coinformation.h"
#include "cpu/information_theory/transfer_entropy.h"
#include "cpu/information_theory/directed_information.h"
#include "cpu/information_theory/active_information_storage.h"
#include "cpu/information_theory/causally_conditioned_entropy.h"
#include "cpu/information_theory/partial_information_decomposition.h"

// Meta backend
#include "meta/information_theory/kullback_leibler_divergence.h"
#include "meta/information_theory/jensen_shannon_divergence.h"
#include "meta/information_theory/shannon_entropy.h"
#include "meta/information_theory/joint_entropy.h"
#include "meta/information_theory/conditional_entropy.h"
#include "meta/information_theory/mutual_information.h"
#include "meta/information_theory/pointwise_mutual_information.h"
#include "meta/information_theory/cross_entropy.h"
#include "meta/information_theory/chi_squared_divergence.h"
#include "meta/information_theory/renyi_entropy.h"
#include "meta/information_theory/tsallis_entropy.h"
#include "meta/information_theory/renyi_divergence.h"
#include "meta/information_theory/conditional_mutual_information.h"
#include "meta/information_theory/total_correlation.h"
#include "meta/information_theory/dual_total_correlation.h"
#include "meta/information_theory/interaction_information.h"
#include "meta/information_theory/coinformation.h"
#include "meta/information_theory/transfer_entropy.h"
#include "meta/information_theory/directed_information.h"
#include "meta/information_theory/active_information_storage.h"
#include "meta/information_theory/causally_conditioned_entropy.h"
#include "meta/information_theory/partial_information_decomposition.h"

// Autograd backend
#include "autograd/information_theory/kullback_leibler_divergence.h"
#include "autograd/information_theory/jensen_shannon_divergence.h"
#include "autograd/information_theory/shannon_entropy.h"
#include "autograd/information_theory/joint_entropy.h"
#include "autograd/information_theory/conditional_entropy.h"
#include "autograd/information_theory/mutual_information.h"
#include "autograd/information_theory/pointwise_mutual_information.h"
#include "autograd/information_theory/cross_entropy.h"
#include "autograd/information_theory/chi_squared_divergence.h"
#include "autograd/information_theory/renyi_entropy.h"
#include "autograd/information_theory/tsallis_entropy.h"
#include "autograd/information_theory/renyi_divergence.h"
#include "autograd/information_theory/conditional_mutual_information.h"
#include "autograd/information_theory/total_correlation.h"
#include "autograd/information_theory/dual_total_correlation.h"
#include "autograd/information_theory/interaction_information.h"
#include "autograd/information_theory/coinformation.h"
#include "autograd/information_theory/transfer_entropy.h"
#include "autograd/information_theory/directed_information.h"
#include "autograd/information_theory/active_information_storage.h"
#include "autograd/information_theory/causally_conditioned_entropy.h"
#include "autograd/information_theory/partial_information_decomposition.h"

TORCH_LIBRARY_FRAGMENT(torchscience, m) {
  // KL divergence
  m.def("kullback_leibler_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  m.def("kullback_leibler_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");
  m.def("kullback_leibler_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor, Tensor)");

  // Jensen-Shannon divergence
  m.def("jensen_shannon_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> Tensor");
  m.def("jensen_shannon_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor)");
  m.def("jensen_shannon_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor, Tensor)");

  // Shannon entropy
  m.def("shannon_entropy(Tensor p, int dim, str input_type, str reduction, float? base) -> Tensor");
  m.def("shannon_entropy_backward(Tensor grad_output, Tensor p, int dim, str input_type, str reduction, float? base) -> Tensor");
  m.def("shannon_entropy_backward_backward(Tensor gg_p, Tensor grad_output, Tensor p, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Joint entropy
  m.def("joint_entropy(Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  m.def("joint_entropy_backward(Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  m.def("joint_entropy_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Conditional entropy
  m.def("conditional_entropy(Tensor joint, int condition_dim, int target_dim, str input_type, str reduction, float? base) -> Tensor");
  m.def("conditional_entropy_backward(Tensor grad_output, Tensor joint, int condition_dim, int target_dim, str input_type, str reduction, float? base) -> Tensor");
  m.def("conditional_entropy_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int condition_dim, int target_dim, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Mutual information
  m.def("mutual_information(Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  m.def("mutual_information_backward(Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  m.def("mutual_information_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Pointwise mutual information
  m.def("pointwise_mutual_information(Tensor joint, int[] dims, str input_type, float? base) -> Tensor");
  m.def("pointwise_mutual_information_backward(Tensor grad_output, Tensor joint, int[] dims, str input_type, float? base) -> Tensor");
  m.def("pointwise_mutual_information_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int[] dims, str input_type, float? base) -> (Tensor, Tensor)");

  // Conditional mutual information
  m.def("conditional_mutual_information(Tensor joint, int[] dims_x, int[] dims_y, int[] dims_z, str input_type, str reduction, float? base) -> Tensor");
  m.def("conditional_mutual_information_backward(Tensor grad_output, Tensor joint, int[] dims_x, int[] dims_y, int[] dims_z, str input_type, str reduction, float? base) -> Tensor");

  // Total correlation (multi-information)
  m.def("total_correlation(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("total_correlation_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Dual total correlation (binding information)
  m.def("dual_total_correlation(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("dual_total_correlation_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Interaction information
  m.def("interaction_information(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("interaction_information_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Coinformation
  m.def("coinformation(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("coinformation_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Transfer entropy
  m.def("transfer_entropy(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("transfer_entropy_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Directed information
  m.def("directed_information(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("directed_information_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Active information storage
  m.def("active_information_storage(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("active_information_storage_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Causally conditioned entropy
  m.def("causally_conditioned_entropy(Tensor joint, str input_type, str reduction, float? base) -> Tensor");
  m.def("causally_conditioned_entropy_backward(Tensor grad_output, Tensor joint, str input_type, str reduction, float? base) -> Tensor");

  // Partial information decomposition
  m.def("partial_information_decomposition(Tensor joint, str method, str input_type, float? base) -> Tensor[]");
  m.def("partial_information_decomposition_backward(Tensor grad_redundancy, Tensor grad_unique_x, Tensor grad_unique_y, Tensor grad_synergy, Tensor grad_mutual_info, Tensor joint, str method, str input_type, float? base) -> Tensor[]");

  // Renyi entropy
  m.def("renyi_entropy(Tensor p, float alpha, int dim, str input_type, str reduction, float? base) -> Tensor");
  m.def("renyi_entropy_backward(Tensor grad_output, Tensor p, float alpha, int dim, str input_type, str reduction, float? base) -> Tensor");
  m.def("renyi_entropy_backward_backward(Tensor gg_p, Tensor grad_output, Tensor p, float alpha, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Tsallis entropy
  m.def("tsallis_entropy(Tensor p, float q, int dim, str input_type, str reduction) -> Tensor");
  m.def("tsallis_entropy_backward(Tensor grad_output, Tensor p, float q, int dim, str input_type, str reduction) -> Tensor");
  m.def("tsallis_entropy_backward_backward(Tensor gg_p, Tensor grad_output, Tensor p, float q, int dim, str input_type, str reduction) -> (Tensor, Tensor)");

  // Renyi divergence
  m.def("renyi_divergence(Tensor p, Tensor q, float alpha, int dim, str input_type, str reduction, float? base, bool pairwise) -> Tensor");
  m.def("renyi_divergence_backward(Tensor grad_output, Tensor p, Tensor q, float alpha, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor)");
  m.def("renyi_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, float alpha, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor, Tensor)");

  // Cross-entropy
  m.def("cross_entropy(Tensor p, Tensor q, int dim, str input_type, str reduction, float? base) -> Tensor");
  m.def("cross_entropy_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor)");
  m.def("cross_entropy_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor, Tensor)");

  // Chi-squared divergence
  m.def("chi_squared_divergence(Tensor p, Tensor q, int dim, str reduction) -> Tensor");
  m.def("chi_squared_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str reduction) -> (Tensor, Tensor)");
  m.def("chi_squared_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str reduction) -> (Tensor, Tensor, Tensor)");
}
