#include <torch/extension.h>

// coding
#include "cpu/coding/morton.h"

// special_functions
#include "cpu/special_functions.h"
#include "meta/special_functions.h"
#include "autograd/special_functions.h"
#include "autocast/special_functions.h"

// combinatorics
#include "cpu/combinatorics.h"
#include "meta/combinatorics.h"
#include "autograd/combinatorics.h"
#include "autocast/combinatorics.h"
#include "sparse/coo/cpu/special_functions.h"
#include "sparse/csr/cpu/special_functions.h"
#include "quantized/cpu/special_functions.h"

// other operators - Phase 2
#include "composite/signal_processing/window_functions.h"
#include "cpu/signal_processing/window_functions.h"
#include "meta/signal_processing/window_functions.h"
#include "autograd/signal_processing/window_functions.h"
#include "cpu/signal_processing/waveform/sine_wave.h"
#include "cpu/signal_processing/waveform/sine_wave_backward.h"
#include "cpu/signal_processing/waveform/square_wave.h"
#include "cpu/signal_processing/waveform/sawtooth_wave.h"
#include "cpu/signal_processing/waveform/triangle_wave.h"
#include "cpu/signal_processing/waveform/pulse_wave.h"
#include "cpu/signal_processing/waveform/impulse_wave.h"
#include "cpu/signal_processing/waveform/step_wave.h"
#include "cpu/signal_processing/waveform/ramp_wave.h"
#include "cpu/signal_processing/waveform/gaussian_pulse_wave.h"
#include "cpu/signal_processing/waveform/sinc_pulse_wave.h"
#include "cpu/signal_processing/waveform/linear_chirp_wave.h"
#include "cpu/signal_processing/waveform/logarithmic_chirp_wave.h"
#include "cpu/signal_processing/waveform/hyperbolic_chirp_wave.h"
#include "cpu/signal_processing/waveform/frequency_modulated_wave.h"
#include "meta/signal_processing/waveform/sine_wave.h"
#include "autograd/signal_processing/waveform/sine_wave.h"
// noise - CompositeExplicitAutograd
#include "cpu/signal_processing/noise/white_noise.h"
#include "cpu/signal_processing/noise/pink_noise.h"
#include "cpu/signal_processing/noise/brown_noise.h"
#include "cpu/signal_processing/noise/blue_noise.h"
#include "cpu/signal_processing/noise/violet_noise.h"
#include "cpu/signal_processing/noise/poisson_noise.h"
#include "cpu/signal_processing/noise/shot_noise.h"
#include "cpu/signal_processing/noise/impulse_noise.h"
#include "composite/optimization/test_functions.h"

#include "cpu/distance/minkowski_distance.h"
#include "cpu/distance/hellinger_distance.h"
#include "cpu/distance/total_variation_distance.h"
#include "cpu/distance/bhattacharyya_distance.h"
#include "cpu/graphics/shading/cook_torrance.h"
#include "cpu/graphics/shading/phong.h"
#include "cpu/graphics/shading/schlick_reflectance.h"
#include "cpu/graphics/lighting/spotlight.h"
#include "cpu/graphics/tone_mapping/reinhard.h"
#include "cpu/graphics/texture_mapping/cube_mapping.h"
#include "cpu/graphics/projection/perspective_projection.h"
#include "cpu/graphics/color/srgb_to_hsv.h"
#include "cpu/graphics/color/hsv_to_srgb.h"
#include "cpu/graphics/color/srgb_to_srgb_linear.h"
#include "cpu/graphics/color/srgb_linear_to_srgb.h"

// morphology
#include "cpu/morphology/erosion.h"
#include "cpu/morphology/dilation.h"
#include "meta/morphology/erosion.h"
#include "meta/morphology/dilation.h"
#include "autograd/morphology/erosion.h"
#include "autograd/morphology/dilation.h"
#include "autocast/morphology/erosion.h"
#include "autocast/morphology/dilation.h"
#include "cpu/signal_processing/filter.h"
#include "cpu/optimization/test_functions.h"
#include "cpu/optimization/combinatorial.h"
#include "cpu/statistics/descriptive/kurtosis.h"
#include "cpu/statistics/descriptive/histogram.h"
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
#include "cpu/integral_transform/hilbert_transform.h"
#include "cpu/integral_transform/inverse_hilbert_transform.h"
#include "cpu/test/sum_squares.h"
#include "cpu/graph_theory/floyd_warshall.h"
#include "cpu/graph_theory/connected_components.h"
#include "cpu/graph_theory/dijkstra.h"
#include "cpu/graph_theory/bellman_ford.h"
#include "cpu/graph_theory/minimum_spanning_tree.h"
#include "cpu/graph_theory/maximum_bipartite_matching.h"
#include "cpu/graph_theory/closeness_centrality.h"
#include "cpu/graph_theory/katz_centrality.h"
#include "cpu/graph_theory/eigenvector_centrality.h"
#include "cpu/graph_theory/betweenness_centrality.h"
#include "cpu/graph_theory/topological_sort.h"
#include "cpu/graph_theory/breadth_first_search.h"
#include "cpu/graph_theory/depth_first_search.h"
#include "cpu/graph_theory/dag_shortest_paths.h"
#include "cpu/graph_theory/edmonds_karp.h"
#include "cpu/graph_theory/push_relabel.h"
#include "cpu/graph_theory/minimum_cut.h"
#include "cpu/graph_theory/min_cost_max_flow.h"
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
#include "cpu/space_partitioning/kd_tree.h"
#include "cpu/space_partitioning/k_nearest_neighbors.h"
#include "cpu/space_partitioning/range_search.h"
#include "cpu/space_partitioning/bvh.h"
#include "cpu/space_partitioning/octree.h"
#include "cpu/geometry/ray_intersect.h"
#include "cpu/geometry/closest_point.h"
#include "cpu/geometry/ray_occluded.h"
#include "cpu/geometry/transform/reflect.h"
#include "cpu/geometry/transform/refract.h"
#include "cpu/geometry/transform/quaternion_multiply.h"
#include "cpu/geometry/transform/quaternion_inverse.h"
#include "cpu/geometry/transform/quaternion_normalize.h"
#include "cpu/geometry/transform/quaternion_apply.h"
#include "cpu/geometry/transform/quaternion_to_matrix.h"
#include "cpu/geometry/transform/matrix_to_quaternion.h"
#include "cpu/geometry/transform/quaternion_slerp.h"
#include "cpu/geometry/convex_hull.h"
#include "cpu/encryption/chacha20.h"
#include "cpu/encryption/sha256.h"
#include "meta/encryption/chacha20.h"
#include "meta/encryption/sha256.h"
#include "cpu/privacy/gaussian_mechanism.h"
#include "cpu/privacy/laplace_mechanism.h"
#include "meta/privacy/gaussian_mechanism.h"
#include "meta/privacy/laplace_mechanism.h"
#include "autograd/privacy/gaussian_mechanism.h"
#include "autograd/privacy/laplace_mechanism.h"

// polynomial
#include "cpu/polynomial/polynomial/evaluate.h"
#include "meta/polynomial/polynomial/evaluate.h"
#include "autograd/polynomial/polynomial/evaluate.h"
#include "autocast/polynomial/polynomial/polynomial_evaluate.h"
#include "autocast/polynomial/polynomial/polynomial_derivative.h"
#include "cpu/polynomial/polynomial/derivative.h"
#include "meta/polynomial/polynomial/derivative.h"
#include "autograd/polynomial/polynomial/polynomial_derivative.h"
#include "cpu/polynomial/polynomial/antiderivative.h"
#include "meta/polynomial/polynomial/antiderivative.h"
#include "autograd/polynomial/polynomial/polynomial_antiderivative.h"
#include "autocast/polynomial/polynomial/polynomial_antiderivative.h"
#include "cpu/polynomial/polynomial/add.h"
#include "meta/polynomial/polynomial/add.h"
#include "autograd/polynomial/polynomial/polynomial_add.h"
#include "autocast/polynomial/polynomial/polynomial_add.h"
#include "cpu/polynomial/polynomial/subtract.h"
#include "meta/polynomial/polynomial/subtract.h"
#include "autograd/polynomial/polynomial/polynomial_subtract.h"
#include "autocast/polynomial/polynomial/polynomial_subtract.h"
#include "cpu/polynomial/polynomial/negate.h"
#include "meta/polynomial/polynomial/negate.h"
#include "autograd/polynomial/polynomial/polynomial_negate.h"
#include "autocast/polynomial/polynomial/polynomial_negate.h"
#include "cpu/polynomial/polynomial/scale.h"
#include "meta/polynomial/polynomial/scale.h"
#include "autograd/polynomial/polynomial/polynomial_scale.h"
#include "autocast/polynomial/polynomial/polynomial_scale.h"
#include "cpu/polynomial/polynomial/multiply.h"
#include "meta/polynomial/polynomial/multiply.h"
#include "autograd/polynomial/polynomial/polynomial_multiply.h"
#include "autocast/polynomial/polynomial/polynomial_multiply.h"
#include "cpu/polynomial/polynomial/divmod.h"
#include "meta/polynomial/polynomial/divmod.h"
#include "autograd/polynomial/polynomial/polynomial_divmod.h"
#include "autocast/polynomial/polynomial/polynomial_divmod.h"
#include "cpu/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_t/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_t/multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_t/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_t/derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_t/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_t/mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_t/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_t/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_t/chebyshev_polynomial_t_antiderivative.h"
#include "cpu/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_u/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_u/multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_u/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_u/derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_u/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_u/mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_u/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_u/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_u/chebyshev_polynomial_u_antiderivative.h"
#include "cpu/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_v/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_v/multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_v/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_v/derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_v/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_v/mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_v/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_v/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_v/chebyshev_polynomial_v_antiderivative.h"
#include "cpu/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "meta/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_evaluate.h"
#include "cpu/polynomial/chebyshev_polynomial_w/multiply.h"
#include "meta/polynomial/chebyshev_polynomial_w/multiply.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_multiply.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_multiply.h"
#include "cpu/polynomial/chebyshev_polynomial_w/derivative.h"
#include "meta/polynomial/chebyshev_polynomial_w/derivative.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_derivative.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_derivative.h"
#include "cpu/polynomial/chebyshev_polynomial_w/mulx.h"
#include "meta/polynomial/chebyshev_polynomial_w/mulx.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_mulx.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_mulx.h"
#include "cpu/polynomial/chebyshev_polynomial_w/antiderivative.h"
#include "meta/polynomial/chebyshev_polynomial_w/antiderivative.h"
#include "autograd/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_antiderivative.h"
#include "autocast/polynomial/chebyshev_polynomial_w/chebyshev_polynomial_w_antiderivative.h"
#include "cpu/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "meta/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_evaluate.h"
#include "cpu/polynomial/legendre_polynomial_p/derivative.h"
#include "meta/polynomial/legendre_polynomial_p/derivative.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_derivative.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_derivative.h"
#include "cpu/polynomial/legendre_polynomial_p/antiderivative.h"
#include "meta/polynomial/legendre_polynomial_p/antiderivative.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_antiderivative.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_antiderivative.h"
#include "cpu/polynomial/legendre_polynomial_p/mulx.h"
#include "meta/polynomial/legendre_polynomial_p/mulx.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_mulx.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_mulx.h"
#include "cpu/polynomial/legendre_polynomial_p/multiply.h"
#include "meta/polynomial/legendre_polynomial_p/multiply.h"
#include "autograd/polynomial/legendre_polynomial_p/legendre_polynomial_p_multiply.h"
#include "autocast/polynomial/legendre_polynomial_p/legendre_polynomial_p_multiply.h"
#include "cpu/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "meta/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_evaluate.h"
#include "cpu/polynomial/laguerre_polynomial_l/derivative.h"
#include "meta/polynomial/laguerre_polynomial_l/derivative.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_derivative.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_derivative.h"
#include "cpu/polynomial/laguerre_polynomial_l/antiderivative.h"
#include "meta/polynomial/laguerre_polynomial_l/antiderivative.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_antiderivative.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_antiderivative.h"
#include "cpu/polynomial/laguerre_polynomial_l/mulx.h"
#include "meta/polynomial/laguerre_polynomial_l/mulx.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_mulx.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_mulx.h"
#include "cpu/polynomial/laguerre_polynomial_l/multiply.h"
#include "meta/polynomial/laguerre_polynomial_l/multiply.h"
#include "autograd/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_multiply.h"
#include "autocast/polynomial/laguerre_polynomial_l/laguerre_polynomial_l_multiply.h"
#include "cpu/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "meta/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_evaluate.h"
#include "cpu/polynomial/hermite_polynomial_h/derivative.h"
#include "meta/polynomial/hermite_polynomial_h/derivative.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_derivative.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_derivative.h"
#include "cpu/polynomial/hermite_polynomial_h/antiderivative.h"
#include "meta/polynomial/hermite_polynomial_h/antiderivative.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_antiderivative.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_antiderivative.h"
#include "cpu/polynomial/hermite_polynomial_h/mulx.h"
#include "meta/polynomial/hermite_polynomial_h/mulx.h"
#include "autograd/polynomial/hermite_polynomial_h/hermite_polynomial_h_mulx.h"
#include "autocast/polynomial/hermite_polynomial_h/hermite_polynomial_h_mulx.h"
#include "cpu/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "meta/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_evaluate.h"
#include "cpu/polynomial/hermite_polynomial_he/derivative.h"
#include "meta/polynomial/hermite_polynomial_he/derivative.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_derivative.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_derivative.h"
#include "cpu/polynomial/hermite_polynomial_he/antiderivative.h"
#include "meta/polynomial/hermite_polynomial_he/antiderivative.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_antiderivative.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_antiderivative.h"
#include "cpu/polynomial/hermite_polynomial_he/mulx.h"
#include "meta/polynomial/hermite_polynomial_he/mulx.h"
#include "autograd/polynomial/hermite_polynomial_he/hermite_polynomial_he_mulx.h"
#include "autocast/polynomial/hermite_polynomial_he/hermite_polynomial_he_mulx.h"
#include "cpu/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "meta/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "autograd/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "autocast/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_evaluate.h"
#include "cpu/polynomial/gegenbauer_polynomial_c/mulx.h"
#include "meta/polynomial/gegenbauer_polynomial_c/mulx.h"
#include "autograd/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_mulx.h"
#include "autocast/polynomial/gegenbauer_polynomial_c/gegenbauer_polynomial_c_mulx.h"
#include "cpu/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "meta/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "autograd/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "autocast/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_evaluate.h"
#include "cpu/polynomial/jacobi_polynomial_p/mulx.h"
#include "meta/polynomial/jacobi_polynomial_p/mulx.h"
#include "autograd/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_mulx.h"
#include "autocast/polynomial/jacobi_polynomial_p/jacobi_polynomial_p_mulx.h"

// linear_algebra decomposition
#include "cpu/linear_algebra/symmetric_generalized_eigenvalue.h"
#include "meta/linear_algebra/symmetric_generalized_eigenvalue.h"
#include "autograd/linear_algebra/symmetric_generalized_eigenvalue.h"
#include "autocast/linear_algebra/symmetric_generalized_eigenvalue.h"
#include "cpu/linear_algebra/generalized_eigenvalue.h"
#include "meta/linear_algebra/generalized_eigenvalue.h"
#include "cpu/linear_algebra/schur_decomposition.h"
#include "meta/linear_algebra/schur_decomposition.h"
#include "cpu/linear_algebra/polar_decomposition.h"
#include "meta/linear_algebra/polar_decomposition.h"
#include "autograd/linear_algebra/polar_decomposition.h"
#include "cpu/linear_algebra/hessenberg.h"
#include "meta/linear_algebra/hessenberg.h"
#include "cpu/linear_algebra/generalized_schur.h"
#include "meta/linear_algebra/generalized_schur.h"
#include "cpu/linear_algebra/jordan_decomposition.h"
#include "meta/linear_algebra/jordan_decomposition.h"
#include "cpu/linear_algebra/pivoted_lu.h"
#include "meta/linear_algebra/pivoted_lu.h"
#include "cpu/linear_algebra/pivoted_qr.h"
#include "meta/linear_algebra/pivoted_qr.h"
#include "cpu/linear_algebra/rank_revealing_qr.h"
#include "meta/linear_algebra/rank_revealing_qr.h"
#include "cpu/linear_algebra/ldl_decomposition.h"
#include "meta/linear_algebra/ldl_decomposition.h"

// pad
#include "cpu/pad/pad.h"
#include "meta/pad/pad.h"
#include "autograd/pad/pad.h"

// probability
#include "cpu/probability/normal.h"
#include "cpu/probability/chi2.h"
#include "cpu/probability/f.h"
#include "cpu/probability/beta.h"
#include "cpu/probability/gamma.h"
#include "cpu/probability/binomial.h"
#include "cpu/probability/poisson.h"
#include "meta/probability/normal.h"
#include "meta/probability/chi2.h"
#include "meta/probability/f.h"
#include "meta/probability/beta.h"
#include "meta/probability/gamma.h"
#include "meta/probability/binomial.h"
#include "meta/probability/poisson.h"
#include "autograd/probability/normal.h"
#include "autograd/probability/chi2.h"
#include "autograd/probability/f.h"
#include "autograd/probability/beta.h"
#include "autograd/probability/gamma.h"
#include "autograd/probability/binomial.h"
#include "autograd/probability/poisson.h"

#include "autograd/distance/minkowski_distance.h"
#include "autograd/distance/hellinger_distance.h"
#include "autograd/distance/total_variation_distance.h"
#include "autograd/distance/bhattacharyya_distance.h"
#include "autograd/graphics/shading/cook_torrance.h"
#include "autograd/graphics/shading/phong.h"
#include "autograd/graphics/shading/schlick_reflectance.h"
#include "autograd/graphics/lighting/spotlight.h"
#include "autograd/graphics/tone_mapping/reinhard.h"
#include "autograd/graphics/projection/perspective_projection.h"
#include "autograd/graphics/color/srgb_to_hsv.h"
#include "autograd/graphics/color/hsv_to_srgb.h"
#include "autograd/graphics/color/srgb_to_srgb_linear.h"
#include "autograd/graphics/color/srgb_linear_to_srgb.h"
#include "autograd/signal_processing/filter.h"
#include "autograd/optimization/test_functions.h"
#include "autograd/optimization/combinatorial.h"
#include "autograd/statistics/descriptive/kurtosis.h"
#include "autograd/statistics/hypothesis_test/jarque_bera.h"
#include "autograd/statistics/hypothesis_test/f_oneway.h"
#include "autograd/statistics/hypothesis_test/chi_square_test.h"
#include "autograd/integral_transform/hilbert_transform.h"
#include "autograd/integral_transform/inverse_hilbert_transform.h"
#include "autograd/test/sum_squares.h"
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
#include "autograd/geometry/transform/reflect.h"
#include "autograd/geometry/transform/refract.h"
#include "autograd/geometry/transform/quaternion_multiply.h"
#include "autograd/geometry/transform/quaternion_inverse.h"
#include "autograd/geometry/transform/quaternion_normalize.h"
#include "autograd/geometry/transform/quaternion_apply.h"
#include "autograd/geometry/transform/quaternion_to_matrix.h"
#include "autograd/geometry/transform/matrix_to_quaternion.h"
#include "autograd/geometry/transform/quaternion_slerp.h"

#include "meta/distance/minkowski_distance.h"
#include "meta/distance/hellinger_distance.h"
#include "meta/distance/total_variation_distance.h"
#include "meta/distance/bhattacharyya_distance.h"
#include "meta/graphics/shading/cook_torrance.h"
#include "meta/graphics/shading/phong.h"
#include "meta/graphics/shading/schlick_reflectance.h"
#include "meta/graphics/lighting/spotlight.h"
#include "meta/graphics/tone_mapping/reinhard.h"
#include "meta/graphics/texture_mapping/cube_mapping.h"
#include "meta/graphics/projection/perspective_projection.h"
#include "meta/graphics/color/srgb_to_hsv.h"
#include "meta/graphics/color/hsv_to_srgb.h"
#include "meta/graphics/color/srgb_to_srgb_linear.h"
#include "meta/graphics/color/srgb_linear_to_srgb.h"
#include "meta/signal_processing/filter.h"
#include "meta/optimization/test_functions.h"
#include "meta/optimization/combinatorial.h"
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
#include "meta/integral_transform/hilbert_transform.h"
#include "meta/integral_transform/inverse_hilbert_transform.h"
#include "meta/test/sum_squares.h"
#include "meta/graph_theory/floyd_warshall.h"
#include "meta/graph_theory/connected_components.h"
#include "meta/graph_theory/dijkstra.h"
#include "meta/graph_theory/bellman_ford.h"
#include "meta/graph_theory/minimum_spanning_tree.h"
#include "meta/graph_theory/maximum_bipartite_matching.h"
#include "meta/graph_theory/closeness_centrality.h"
#include "meta/graph_theory/katz_centrality.h"
#include "meta/graph_theory/eigenvector_centrality.h"
#include "meta/graph_theory/betweenness_centrality.h"
#include "meta/graph_theory/topological_sort.h"
#include "meta/graph_theory/breadth_first_search.h"
#include "meta/graph_theory/depth_first_search.h"
#include "meta/graph_theory/dag_shortest_paths.h"
#include "meta/graph_theory/edmonds_karp.h"
#include "meta/graph_theory/push_relabel.h"
#include "meta/graph_theory/minimum_cut.h"
#include "meta/graph_theory/min_cost_max_flow.h"
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
#include "meta/space_partitioning/kd_tree.h"
#include "meta/space_partitioning/k_nearest_neighbors.h"
#include "meta/space_partitioning/range_search.h"
#include "meta/geometry/transform/reflect.h"
#include "meta/geometry/transform/refract.h"
#include "meta/geometry/transform/quaternion_multiply.h"
#include "meta/geometry/transform/quaternion_inverse.h"
#include "meta/geometry/transform/quaternion_normalize.h"
#include "meta/geometry/transform/quaternion_apply.h"
#include "meta/geometry/transform/quaternion_to_matrix.h"
#include "meta/geometry/transform/matrix_to_quaternion.h"
#include "meta/geometry/transform/quaternion_slerp.h"
#include "meta/geometry/convex_hull.h"
#include "autograd/space_partitioning/k_nearest_neighbors.h"
#include "autograd/space_partitioning/range_search.h"
#include "autograd/space_partitioning/octree.h"

#include "autocast/signal_processing/filter.h"
#include "autocast/statistics/descriptive/kurtosis.h"
#include "autocast/integral_transform/hilbert_transform.h"
#include "autocast/integral_transform/inverse_hilbert_transform.h"
#include "autocast/test/sum_squares.h"
#include "autocast/space_partitioning/kd_tree.h"
#include "autocast/space_partitioning/k_nearest_neighbors.h"
#include "autocast/space_partitioning/range_search.h"
#include "autocast/pad/pad.h"

#include "sparse/coo/cpu/optimization/test_functions.h"
#include "sparse/coo/cpu/integral_transform/hilbert_transform.h"
#include "sparse/coo/cpu/integral_transform/inverse_hilbert_transform.h"
#include "sparse/csr/cpu/optimization/test_functions.h"
#include "sparse/csr/cpu/integral_transform/hilbert_transform.h"
#include "sparse/csr/cpu/integral_transform/inverse_hilbert_transform.h"
#include "quantized/cpu/optimization/test_functions.h"
#include "quantized/cpu/integral_transform/hilbert_transform.h"
#include "quantized/cpu/integral_transform/inverse_hilbert_transform.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/graphics/shading/cook_torrance.cu"
#include "cuda/optimization/test_functions.cu"
#include "cuda/statistics/descriptive/kurtosis.cu"
#include "cuda/statistics/descriptive/histogram.cu"
#include "cuda/integral_transform/hilbert_transform.cu"
#include "cuda/integral_transform/inverse_hilbert_transform.cu"
#include "cuda/graph_theory/floyd_warshall.cu"
#include "sparse/coo/cuda/special_functions.h"
#include "sparse/coo/cuda/optimization/test_functions.h"
#include "sparse/coo/cuda/integral_transform/hilbert_transform.h"
#include "sparse/coo/cuda/integral_transform/inverse_hilbert_transform.h"
#include "sparse/csr/cuda/special_functions.h"
#include "sparse/csr/cuda/optimization/test_functions.h"
#include "sparse/csr/cuda/integral_transform/hilbert_transform.h"
#include "sparse/csr/cuda/integral_transform/inverse_hilbert_transform.h"
#include "quantized/cuda/special_functions.h"
#include "quantized/cuda/optimization/test_functions.h"
#include "quantized/cuda/integral_transform/hilbert_transform.h"
#include "quantized/cuda/integral_transform/inverse_hilbert_transform.h"
#include "cuda/space_partitioning/kd_tree.cuh"
#endif

extern "C" {
  PyObject* PyInit__csrc(void) {
    static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_csrc",
      nullptr,
      -1,
      nullptr,
    };

    return PyModule_Create(&module_def);
  }
}

TORCH_LIBRARY(torchscience, module) {
  // special_functions
  module.def("gamma(Tensor z) -> Tensor");
  module.def("gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("gamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  module.def("digamma(Tensor z) -> Tensor");
  module.def("digamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("digamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  module.def("trigamma(Tensor z) -> Tensor");
  module.def("trigamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("trigamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  module.def("beta(Tensor a, Tensor b) -> Tensor");
  module.def("beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  module.def("chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor");
  module.def("chebyshev_polynomial_t_backward(Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_t_backward_backward(Tensor gg_x, Tensor gg_n, Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor, Tensor)");

  module.def("incomplete_beta(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("incomplete_beta_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("incomplete_beta_backward_backward(Tensor gg_x, Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  module.def("hypergeometric_2_f_1(Tensor a, Tensor b, Tensor c, Tensor z) -> Tensor");
  module.def("hypergeometric_2_f_1_backward(Tensor grad_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("hypergeometric_2_f_1_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_c, Tensor gg_z, Tensor grad_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  module.def("polygamma(Tensor n, Tensor z) -> Tensor");
  module.def("polygamma_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("polygamma_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  module.def("log_beta(Tensor a, Tensor b) -> Tensor");
  module.def("log_beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("log_beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  module.def("log_gamma(Tensor z) -> Tensor");
  module.def("log_gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("log_gamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  module.def("regularized_gamma_p(Tensor a, Tensor x) -> Tensor");
  module.def("regularized_gamma_p_backward(Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor)");
  module.def("regularized_gamma_p_backward_backward(Tensor grad_grad_a, Tensor grad_grad_x, Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor, Tensor)");

  module.def("regularized_gamma_q(Tensor a, Tensor x) -> Tensor");
  module.def("regularized_gamma_q_backward(Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor)");
  module.def("regularized_gamma_q_backward_backward(Tensor grad_grad_a, Tensor grad_grad_x, Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Modified Bessel I₀
  module.def("modified_bessel_i_0(Tensor z) -> Tensor");
  module.def("modified_bessel_i_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("modified_bessel_i_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified Bessel I₁
  module.def("modified_bessel_i_1(Tensor z) -> Tensor");
  module.def("modified_bessel_i_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("modified_bessel_i_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

// Bessel J₀
  module.def("bessel_j_0(Tensor z) -> Tensor");
  module.def("bessel_j_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("bessel_j_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Bessel J₁
  module.def("bessel_j_1(Tensor z) -> Tensor");
  module.def("bessel_j_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("bessel_j_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Bessel Y₀
  module.def("bessel_y_0(Tensor z) -> Tensor");
  module.def("bessel_y_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("bessel_y_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Bessel Y₁
  module.def("bessel_y_1(Tensor z) -> Tensor");
  module.def("bessel_y_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("bessel_y_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified Bessel K₀
  module.def("modified_bessel_k_0(Tensor z) -> Tensor");
  module.def("modified_bessel_k_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("modified_bessel_k_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified Bessel K₁
  module.def("modified_bessel_k_1(Tensor z) -> Tensor");
  module.def("modified_bessel_k_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("modified_bessel_k_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Bessel Jₙ (general order)
  module.def("bessel_j(Tensor n, Tensor z) -> Tensor");
  module.def("bessel_j_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("bessel_j_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Bessel Yₙ (general order)
  module.def("bessel_y(Tensor n, Tensor z) -> Tensor");
  module.def("bessel_y_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("bessel_y_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Modified Bessel Kₙ (general order)
  module.def("modified_bessel_k(Tensor n, Tensor z) -> Tensor");
  module.def("modified_bessel_k_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("modified_bessel_k_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Modified Bessel Iₙ (general order)
  module.def("modified_bessel_i(Tensor n, Tensor z) -> Tensor");
  module.def("modified_bessel_i_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("modified_bessel_i_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Spherical Bessel j₀
  module.def("spherical_bessel_j_0(Tensor z) -> Tensor");
  module.def("spherical_bessel_j_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_j_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Spherical Bessel j₁
  module.def("spherical_bessel_j_1(Tensor z) -> Tensor");
  module.def("spherical_bessel_j_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_j_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Spherical Bessel jₙ (general order)
  module.def("spherical_bessel_j(Tensor n, Tensor z) -> Tensor");
  module.def("spherical_bessel_j_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("spherical_bessel_j_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Spherical Bessel y₀
  module.def("spherical_bessel_y_0(Tensor z) -> Tensor");
  module.def("spherical_bessel_y_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_y_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Spherical Bessel y₁
  module.def("spherical_bessel_y_1(Tensor z) -> Tensor");
  module.def("spherical_bessel_y_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_y_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Spherical Bessel yₙ (general order)
  module.def("spherical_bessel_y(Tensor n, Tensor z) -> Tensor");
  module.def("spherical_bessel_y_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("spherical_bessel_y_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Modified spherical Bessel i₀
  module.def("spherical_bessel_i_0(Tensor z) -> Tensor");
  module.def("spherical_bessel_i_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_i_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified spherical Bessel i₁
  module.def("spherical_bessel_i_1(Tensor z) -> Tensor");
  module.def("spherical_bessel_i_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_i_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified spherical Bessel iₙ (general order)
  module.def("spherical_bessel_i(Tensor n, Tensor z) -> Tensor");
  module.def("spherical_bessel_i_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("spherical_bessel_i_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Modified spherical Bessel k₀
  module.def("spherical_bessel_k_0(Tensor z) -> Tensor");
  module.def("spherical_bessel_k_0_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_k_0_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified spherical Bessel k₁
  module.def("spherical_bessel_k_1(Tensor z) -> Tensor");
  module.def("spherical_bessel_k_1_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("spherical_bessel_k_1_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // Modified spherical Bessel k (general order)
  module.def("spherical_bessel_k(Tensor n, Tensor z) -> Tensor");
  module.def("spherical_bessel_k_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("spherical_bessel_k_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  // Carlson elliptic integrals
  module.def("carlson_elliptic_integral_r_f(Tensor x, Tensor y, Tensor z) -> Tensor");
  module.def("carlson_elliptic_integral_r_f_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_f_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  module.def("carlson_elliptic_integral_r_d(Tensor x, Tensor y, Tensor z) -> Tensor");
  module.def("carlson_elliptic_integral_r_d_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_d_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  module.def("carlson_elliptic_integral_r_c(Tensor x, Tensor y) -> Tensor");
  module.def("carlson_elliptic_integral_r_c_backward(Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_c_backward_backward(Tensor gg_x, Tensor gg_y, Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor, Tensor)");

  module.def("carlson_elliptic_integral_r_j(Tensor x, Tensor y, Tensor z, Tensor p) -> Tensor");
  module.def("carlson_elliptic_integral_r_j_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z, Tensor p) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_j_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor gg_p, Tensor grad_output, Tensor x, Tensor y, Tensor z, Tensor p) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  module.def("carlson_elliptic_integral_r_g(Tensor x, Tensor y, Tensor z) -> Tensor");
  module.def("carlson_elliptic_integral_r_g_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_g_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  module.def("carlson_elliptic_integral_r_e(Tensor x, Tensor y, Tensor z) -> Tensor");
  module.def("carlson_elliptic_integral_r_e_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_e_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  module.def("carlson_elliptic_integral_r_m(Tensor x, Tensor y, Tensor z) -> Tensor");
  module.def("carlson_elliptic_integral_r_m_backward(Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_m_backward_backward(Tensor gg_x, Tensor gg_y, Tensor gg_z, Tensor grad_output, Tensor x, Tensor y, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");

  module.def("carlson_elliptic_integral_r_k(Tensor x, Tensor y) -> Tensor");
  module.def("carlson_elliptic_integral_r_k_backward(Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor)");
  module.def("carlson_elliptic_integral_r_k_backward_backward(Tensor gg_x, Tensor gg_y, Tensor grad_output, Tensor x, Tensor y) -> (Tensor, Tensor, Tensor)");

  // Legendre elliptic integrals
  module.def("complete_legendre_elliptic_integral_k(Tensor m) -> Tensor");
  module.def("complete_legendre_elliptic_integral_k_backward(Tensor grad_output, Tensor m) -> Tensor");
  module.def("complete_legendre_elliptic_integral_k_backward_backward(Tensor gg_m, Tensor grad_output, Tensor m) -> (Tensor, Tensor)");

  module.def("complete_legendre_elliptic_integral_e(Tensor m) -> Tensor");
  module.def("complete_legendre_elliptic_integral_e_backward(Tensor grad_output, Tensor m) -> Tensor");
  module.def("complete_legendre_elliptic_integral_e_backward_backward(Tensor gg_m, Tensor grad_output, Tensor m) -> (Tensor, Tensor)");

  module.def("incomplete_legendre_elliptic_integral_e(Tensor phi, Tensor m) -> Tensor");
  module.def("incomplete_legendre_elliptic_integral_e_backward(Tensor grad_output, Tensor phi, Tensor m) -> (Tensor, Tensor)");
  module.def("incomplete_legendre_elliptic_integral_e_backward_backward(Tensor gg_phi, Tensor gg_m, Tensor grad_output, Tensor phi, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("complete_legendre_elliptic_integral_pi(Tensor n, Tensor m) -> Tensor");
  module.def("complete_legendre_elliptic_integral_pi_backward(Tensor grad_output, Tensor n, Tensor m) -> (Tensor, Tensor)");
  module.def("complete_legendre_elliptic_integral_pi_backward_backward(Tensor gg_n, Tensor gg_m, Tensor grad_output, Tensor n, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("incomplete_legendre_elliptic_integral_pi(Tensor n, Tensor phi, Tensor m) -> Tensor");
  module.def("incomplete_legendre_elliptic_integral_pi_backward(Tensor grad_output, Tensor n, Tensor phi, Tensor m) -> (Tensor, Tensor, Tensor)");
  module.def("incomplete_legendre_elliptic_integral_pi_backward_backward(Tensor gg_n, Tensor gg_phi, Tensor gg_m, Tensor grad_output, Tensor n, Tensor phi, Tensor m) -> (Tensor, Tensor, Tensor, Tensor)");

  // Jacobi elliptic functions
  module.def("jacobi_amplitude_am(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_amplitude_am_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_amplitude_am_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_dn(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_dn_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_dn_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_cn(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_cn_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_cn_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_sn(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_sn_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_sn_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_sd(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_sd_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_sd_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_cd(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_cd_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_cd_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_sc(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_sc_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_sc_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_nd(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_nd_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_nd_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_nc(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_nc_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_nc_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_ns(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_ns_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_ns_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_dc(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_dc_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_dc_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_ds(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_ds_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_ds_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("jacobi_elliptic_cs(Tensor u, Tensor m) -> Tensor");
  module.def("jacobi_elliptic_cs_backward(Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor)");
  module.def("jacobi_elliptic_cs_backward_backward(Tensor gg_u, Tensor gg_m, Tensor grad_output, Tensor u, Tensor m) -> (Tensor, Tensor, Tensor)");

  // Inverse Jacobi elliptic functions (primary)
  module.def("inverse_jacobi_elliptic_sn(Tensor x, Tensor m) -> Tensor");
  module.def("inverse_jacobi_elliptic_sn_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  module.def("inverse_jacobi_elliptic_sn_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("inverse_jacobi_elliptic_cn(Tensor x, Tensor m) -> Tensor");
  module.def("inverse_jacobi_elliptic_cn_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  module.def("inverse_jacobi_elliptic_cn_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("inverse_jacobi_elliptic_dn(Tensor x, Tensor m) -> Tensor");
  module.def("inverse_jacobi_elliptic_dn_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  module.def("inverse_jacobi_elliptic_dn_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  // Inverse Jacobi elliptic functions (derived)
  module.def("inverse_jacobi_elliptic_sd(Tensor x, Tensor m) -> Tensor");
  module.def("inverse_jacobi_elliptic_sd_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  module.def("inverse_jacobi_elliptic_sd_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("inverse_jacobi_elliptic_cd(Tensor x, Tensor m) -> Tensor");
  module.def("inverse_jacobi_elliptic_cd_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  module.def("inverse_jacobi_elliptic_cd_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  module.def("inverse_jacobi_elliptic_sc(Tensor x, Tensor m) -> Tensor");
  module.def("inverse_jacobi_elliptic_sc_backward(Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor)");
  module.def("inverse_jacobi_elliptic_sc_backward_backward(Tensor gg_x, Tensor gg_m, Tensor grad_output, Tensor x, Tensor m) -> (Tensor, Tensor, Tensor)");

  // Jacobi theta functions
  module.def("theta_1(Tensor z, Tensor q) -> Tensor");
  module.def("theta_1_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  module.def("theta_1_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  module.def("theta_2(Tensor z, Tensor q) -> Tensor");
  module.def("theta_2_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  module.def("theta_2_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  module.def("theta_3(Tensor z, Tensor q) -> Tensor");
  module.def("theta_3_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  module.def("theta_3_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  module.def("theta_4(Tensor z, Tensor q) -> Tensor");
  module.def("theta_4_backward(Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor)");
  module.def("theta_4_backward_backward(Tensor gg_z, Tensor gg_q, Tensor grad_output, Tensor z, Tensor q) -> (Tensor, Tensor, Tensor)");

  // distance
  module.def("minkowski_distance(Tensor x, Tensor y, float p, Tensor? weight) -> Tensor");
  module.def("minkowski_distance_backward(Tensor grad_output, Tensor x, Tensor y, float p, Tensor? weight, Tensor dist_output) -> (Tensor, Tensor, Tensor)");

  // Hellinger distance
  module.def("hellinger_distance(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  module.def("hellinger_distance_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");

  // Total variation distance
  module.def("total_variation_distance(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  module.def("total_variation_distance_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");

  // Bhattacharyya distance
  module.def("bhattacharyya_distance(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  module.def("bhattacharyya_distance_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");

  // graphics.shading
  module.def("cook_torrance(Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> Tensor");
  module.def("cook_torrance_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def("cook_torrance_backward_backward(Tensor gg_normal, Tensor gg_view, Tensor gg_light, Tensor gg_roughness, Tensor gg_f0, Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Phong shading
  module.def("phong(Tensor normal, Tensor view, Tensor light, Tensor shininess) -> Tensor");
  module.def("phong_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor shininess) -> (Tensor, Tensor, Tensor, Tensor)");

  // Schlick reflectance (Fresnel approximation)
  module.def("schlick_reflectance(Tensor cosine, Tensor r0) -> Tensor");
  module.def("schlick_reflectance_backward(Tensor grad_output, Tensor cosine, Tensor r0) -> Tensor");

  // graphics.lighting
  module.def("spotlight(Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor)");
  module.def("spotlight_backward(Tensor grad_irradiance, Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // graphics.tone_mapping
  module.def("reinhard(Tensor input, Tensor? white_point) -> Tensor");
  module.def("reinhard_backward(Tensor grad_output, Tensor input, Tensor? white_point) -> (Tensor, Tensor)");

  // graphics.color
  module.def("srgb_to_hsv(Tensor input) -> Tensor");
  module.def("srgb_to_hsv_backward(Tensor grad_output, Tensor input) -> Tensor");

  module.def("hsv_to_srgb(Tensor input) -> Tensor");
  module.def("hsv_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  module.def("srgb_to_srgb_linear(Tensor input) -> Tensor");
  module.def("srgb_to_srgb_linear_backward(Tensor grad_output, Tensor input) -> Tensor");

  module.def("srgb_linear_to_srgb(Tensor input) -> Tensor");
  module.def("srgb_linear_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  // graphics.texture_mapping
  module.def("cube_mapping(Tensor direction) -> (Tensor, Tensor, Tensor)");

  // graphics.projection
  module.def("perspective_projection(Tensor fov, Tensor aspect, Tensor near, Tensor far) -> Tensor");
  module.def("perspective_projection_backward(Tensor grad_output, Tensor fov, Tensor aspect, Tensor near, Tensor far) -> (Tensor, Tensor, Tensor, Tensor)");

  // optimization.test_functions
  module.def("rosenbrock(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("rosenbrock_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("rosenbrock_backward_backward(Tensor gg_x, Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // optimization.combinatorial
  module.def("sinkhorn(Tensor C, Tensor a, Tensor b, float epsilon, int maxiter, float tol) -> Tensor");
  module.def("sinkhorn_backward(Tensor grad_output, Tensor P, Tensor C, float epsilon) -> Tensor");

  // signal_processing.filter
  module.def("butterworth_analog_bandpass_filter(int n, Tensor omega_p1, Tensor omega_p2) -> Tensor");
  module.def("butterworth_analog_bandpass_filter_backward(Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor)");
  module.def("butterworth_analog_bandpass_filter_backward_backward(Tensor gg_omega_p1, Tensor gg_omega_p2, Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor, Tensor)");

  // signal_processing.waveform
  module.def("sine_wave(int? n=None, Tensor? t=None, *, "
             "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");
  module.def("sine_wave_backward(Tensor grad_output, int? n, Tensor? t, "
             "Tensor frequency, float sample_rate, Tensor amplitude, Tensor phase) -> "
             "(Tensor, Tensor, Tensor, Tensor)");

  module.def("square_wave(int? n=None, Tensor? t=None, *, "
             "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, Tensor duty, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("sawtooth_wave(int? n=None, Tensor? t=None, *, "
             "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("triangle_wave(int? n=None, Tensor? t=None, *, "
             "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("pulse_wave(int? n=None, Tensor? t=None, *, "
             "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, Tensor duty_cycle, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("impulse_wave(int n, *, "
             "Tensor position, Tensor amplitude, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("step_wave(int n, *, "
             "Tensor position, Tensor amplitude, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("ramp_wave(int n, *, "
             "Tensor position, Tensor slope, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("gaussian_pulse_wave(int n, *, "
             "Tensor center, Tensor std, Tensor amplitude, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("sinc_pulse_wave(int n, *, "
             "Tensor center, Tensor bandwidth, Tensor amplitude, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("linear_chirp_wave(int? n=None, Tensor? t=None, *, "
             "Tensor f0, Tensor f1, float t1=1.0, float sample_rate=1.0, "
             "Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("logarithmic_chirp_wave(int? n=None, Tensor? t=None, *, "
             "Tensor f0, Tensor f1, float t1=1.0, float sample_rate=1.0, "
             "Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  module.def("hyperbolic_chirp_wave(int? n=None, Tensor? t=None, *, "
             "Tensor f0, Tensor f1, float t1=1.0, float sample_rate=1.0, "
             "Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  // Sinusoidal FM
  module.def("frequency_modulated_wave(int? n=None, Tensor? t=None, *, "
             "Tensor carrier_frequency, Tensor modulator_frequency, Tensor modulation_index, "
             "float sample_rate=1.0, Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  // Arbitrary modulating signal FM - different name to avoid overload issues
  module.def("frequency_modulated_wave_arbitrary(int? n=None, Tensor? t=None, *, "
             "Tensor carrier_frequency, Tensor modulating_signal, Tensor modulation_index, "
             "float sample_rate=1.0, Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  // signal_processing.window_function
  module.def("rectangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");

  // Parameterless windows
  module.def("hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("hamming_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_hamming_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("blackman_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_blackman_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("bartlett_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_bartlett_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("cosine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_cosine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("nuttall_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_nuttall_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("triangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_triangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("welch_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_welch_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("parzen_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_parzen_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("blackman_harris_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_blackman_harris_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("flat_top_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_flat_top_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("sine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_sine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("bartlett_hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_bartlett_hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("lanczos_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_lanczos_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");

  // Parameterized windows: Gaussian
  module.def("gaussian_window(int n, Tensor std, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_gaussian_window(int n, Tensor std, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("gaussian_window_backward(Tensor grad_output, Tensor output, int n, Tensor std) -> Tensor");
  module.def("periodic_gaussian_window_backward(Tensor grad_output, Tensor output, int n, Tensor std) -> Tensor");

  // Parameterized windows: General Hamming
  module.def("general_hamming_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_general_hamming_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("general_hamming_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");
  module.def("periodic_general_hamming_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");

  // Parameterized windows: General Cosine
  module.def("general_cosine_window(int n, Tensor coeffs, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_general_cosine_window(int n, Tensor coeffs, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("general_cosine_window_backward(Tensor grad_output, Tensor output, int n, Tensor coeffs) -> Tensor");
  module.def("periodic_general_cosine_window_backward(Tensor grad_output, Tensor output, int n, Tensor coeffs) -> Tensor");

  // Parameterized windows: Tukey
  module.def("tukey_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_tukey_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("tukey_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");
  module.def("periodic_tukey_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");

  // Parameterized windows: Exponential
  module.def("exponential_window(int n, Tensor tau, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_exponential_window(int n, Tensor tau, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("exponential_window_backward(Tensor grad_output, Tensor output, int n, Tensor tau) -> Tensor");
  module.def("periodic_exponential_window_backward(Tensor grad_output, Tensor output, int n, Tensor tau) -> Tensor");

  // Parameterized windows: Hann-Poisson
  module.def("hann_poisson_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_hann_poisson_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("hann_poisson_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");
  module.def("periodic_hann_poisson_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");

  // Parameterized windows: Generalized Normal (two parameters)
  module.def("generalized_normal_window(int n, Tensor p, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_generalized_normal_window(int n, Tensor p, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("generalized_normal_window_backward(Tensor grad_output, Tensor output, int n, Tensor p, Tensor sigma) -> (Tensor, Tensor)");
  module.def("periodic_generalized_normal_window_backward(Tensor grad_output, Tensor output, int n, Tensor p, Tensor sigma) -> (Tensor, Tensor)");

  // Parameterized windows: Kaiser
  module.def("kaiser_window(int n, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_kaiser_window(int n, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("kaiser_window_backward(Tensor grad_output, Tensor output, int n, Tensor beta) -> Tensor");
  module.def("periodic_kaiser_window_backward(Tensor grad_output, Tensor output, int n, Tensor beta) -> Tensor");

  // Parameterized windows: Planck-taper
  module.def("planck_taper_window(int n, Tensor epsilon, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_planck_taper_window(int n, Tensor epsilon, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("planck_taper_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon) -> Tensor");
  module.def("periodic_planck_taper_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon) -> Tensor");

  // Parameterized windows: Planck-Bessel (two parameters)
  module.def("planck_bessel_window(int n, Tensor epsilon, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_planck_bessel_window(int n, Tensor epsilon, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("planck_bessel_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon, Tensor beta) -> (Tensor, Tensor)");
  module.def("periodic_planck_bessel_window_backward(Tensor grad_output, Tensor output, int n, Tensor epsilon, Tensor beta) -> (Tensor, Tensor)");

  // FFT-based windows: Dolph-Chebyshev (no explicit backward - uses CompositeImplicitAutograd)
  module.def("dolph_chebyshev_window(int n, Tensor attenuation, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_dolph_chebyshev_window(int n, Tensor attenuation, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");

  // FFT-based windows: Ultraspherical (two parameters, no explicit backward)
  module.def("ultraspherical_window(int n, Tensor mu, Tensor x_mu, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_ultraspherical_window(int n, Tensor mu, Tensor x_mu, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");

  // Phase 5: Eigenvalue-based and polynomial windows (CompositeImplicitAutograd)
  module.def("discrete_prolate_spheroidal_sequence_window(int n, Tensor nw, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_discrete_prolate_spheroidal_sequence_window(int n, Tensor nw, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("approximate_confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_approximate_confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_confined_gaussian_window(int n, Tensor sigma, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("generalized_adaptive_polynomial_window(int n, Tensor alpha, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_generalized_adaptive_polynomial_window(int n, Tensor alpha, Tensor beta, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");

  // statistics.descriptive
  module.def("kurtosis(Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> (Tensor, Tensor)");

  module.def("histogram(Tensor input, int bins, float[]? range, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");
  module.def("histogram_edges(Tensor input, Tensor bins, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");

  // statistics.hypothesis_test
  module.def("one_sample_t_test(Tensor input, float popmean, str alternative) -> (Tensor, Tensor, Tensor)");
  module.def("two_sample_t_test(Tensor input1, Tensor input2, bool equal_var, str alternative) -> (Tensor, Tensor, Tensor)");
  module.def("paired_t_test(Tensor input1, Tensor input2, str alternative) -> (Tensor, Tensor, Tensor)");
  module.def("shapiro_wilk(Tensor input) -> (Tensor, Tensor)");
  module.def("anderson_darling(Tensor input) -> (Tensor, Tensor, Tensor)");
  module.def("f_oneway(Tensor data, Tensor group_sizes) -> (Tensor, Tensor)");
  module.def("f_oneway_backward(Tensor grad_statistic, Tensor data, Tensor group_sizes) -> Tensor");
  module.def("jarque_bera(Tensor input) -> (Tensor, Tensor)");
  module.def("jarque_bera_backward(Tensor grad_statistic, Tensor input) -> Tensor");
  module.def("chi_square_test(Tensor observed, Tensor? expected, int ddof) -> (Tensor, Tensor)");
  module.def("chi_square_test_backward(Tensor grad_statistic, Tensor observed, Tensor? expected) -> Tensor");

  // Non-parametric rank-based tests (no gradients)
  module.def("mann_whitney_u(Tensor x, Tensor y, str alternative) -> (Tensor, Tensor)");
  module.def("wilcoxon_signed_rank(Tensor x, Tensor? y, str alternative, str zero_method) -> (Tensor, Tensor)");
  module.def("kruskal_wallis(Tensor data, Tensor group_sizes) -> (Tensor, Tensor)");

  // integral_transform
  module.def("hilbert_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("hilbert_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("hilbert_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

  module.def("inverse_hilbert_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("inverse_hilbert_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("inverse_hilbert_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

  // test (for validating reduction macros)
  module.def("sum_squares(Tensor input, int[]? dim, bool keepdim) -> Tensor");
  module.def("sum_squares_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> Tensor");
  module.def("sum_squares_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> (Tensor, Tensor)");

  // graph_theory
  module.def("floyd_warshall(Tensor input, bool directed) -> (Tensor, Tensor, bool)");
  module.def("connected_components(Tensor adjacency, bool directed, str connection) -> (int, Tensor)");
  module.def("dijkstra(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor)");
  module.def("bellman_ford(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor, bool)");
  module.def("minimum_spanning_tree(Tensor adjacency) -> (Tensor, Tensor)");
  module.def("maximum_bipartite_matching(Tensor biadjacency) -> (Tensor, Tensor, Tensor)");
  module.def("closeness_centrality(Tensor adjacency, bool normalized) -> Tensor");
  module.def("closeness_centrality_backward(Tensor grad, Tensor adjacency, Tensor distances, bool normalized) -> Tensor");
  module.def("katz_centrality(Tensor adjacency, float alpha, float beta, bool normalized) -> Tensor");
  module.def("eigenvector_centrality(Tensor adjacency) -> Tensor");
  module.def("betweenness_centrality(Tensor adjacency, bool normalized) -> Tensor");
  module.def("topological_sort(Tensor adjacency) -> Tensor");
  module.def("breadth_first_search(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor)");
  module.def("depth_first_search(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor, Tensor)");
  module.def("dag_shortest_paths(Tensor adjacency, int source) -> (Tensor, Tensor)");
  module.def("edmonds_karp(Tensor capacity, int source, int sink) -> (Tensor, Tensor)");
  module.def("push_relabel(Tensor capacity, int source, int sink) -> (Tensor, Tensor)");
  module.def("minimum_cut(Tensor capacity, int source, int sink) -> (Tensor, Tensor, Tensor)");
  module.def("min_cost_max_flow(Tensor capacity, Tensor cost, int source, int sink) -> (Tensor, Tensor, Tensor)");

  // combinatorics
  module.def("binomial_coefficient(Tensor n, Tensor k) -> Tensor");
  module.def("binomial_coefficient_backward(Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor)");
  module.def("binomial_coefficient_backward_backward(Tensor gg_n, Tensor gg_k, Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor, Tensor)");

  // signal_processing.noise
  module.def("white_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("pink_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("brown_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("blue_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("violet_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("poisson_noise(int[] size, Tensor rate, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, Generator? generator=None) -> Tensor");
  module.def("shot_noise(int[] size, Tensor rate, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("impulse_noise(int[] size, Tensor p_salt, Tensor p_pepper, float salt_value, float pepper_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, Generator? generator=None) -> Tensor");

  // information_theory
  module.def("kullback_leibler_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  module.def("kullback_leibler_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");
  module.def("kullback_leibler_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor, Tensor)");

  module.def("jensen_shannon_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> Tensor");
  module.def("jensen_shannon_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor)");
  module.def("jensen_shannon_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor, Tensor)");

  // Shannon entropy
  module.def("shannon_entropy(Tensor p, int dim, str input_type, str reduction, float? base) -> Tensor");
  module.def("shannon_entropy_backward(Tensor grad_output, Tensor p, int dim, str input_type, str reduction, float? base) -> Tensor");
  module.def("shannon_entropy_backward_backward(Tensor gg_p, Tensor grad_output, Tensor p, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Joint entropy
  module.def("joint_entropy(Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  module.def("joint_entropy_backward(Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  module.def("joint_entropy_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Conditional entropy
  module.def("conditional_entropy(Tensor joint, int condition_dim, int target_dim, str input_type, str reduction, float? base) -> Tensor");
  module.def("conditional_entropy_backward(Tensor grad_output, Tensor joint, int condition_dim, int target_dim, str input_type, str reduction, float? base) -> Tensor");
  module.def("conditional_entropy_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int condition_dim, int target_dim, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Mutual information
  module.def("mutual_information(Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  module.def("mutual_information_backward(Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> Tensor");
  module.def("mutual_information_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int[] dims, str input_type, str reduction, float? base) -> (Tensor, Tensor)");

  // Pointwise mutual information
  module.def("pointwise_mutual_information(Tensor joint, int[] dims, str input_type, float? base) -> Tensor");
  module.def("pointwise_mutual_information_backward(Tensor grad_output, Tensor joint, int[] dims, str input_type, float? base) -> Tensor");
  module.def("pointwise_mutual_information_backward_backward(Tensor gg_joint, Tensor grad_output, Tensor joint, int[] dims, str input_type, float? base) -> (Tensor, Tensor)");

  // Renyi entropy
  module.def("renyi_entropy(Tensor p, float alpha, int dim, str input_type, str reduction, float? base) -> Tensor");
  module.def("renyi_entropy_backward(Tensor grad_output, Tensor p, float alpha, int dim, str input_type, str reduction, float? base) -> Tensor");

  // Tsallis entropy
  module.def("tsallis_entropy(Tensor p, float q, int dim, str input_type, str reduction) -> Tensor");
  module.def("tsallis_entropy_backward(Tensor grad_output, Tensor p, float q, int dim, str input_type, str reduction) -> Tensor");

  // Renyi divergence
  module.def("renyi_divergence(Tensor p, Tensor q, float alpha, int dim, str input_type, str reduction, float? base, bool pairwise) -> Tensor");
  module.def("renyi_divergence_backward(Tensor grad_output, Tensor p, Tensor q, float alpha, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor)");

  // Cross-entropy
  module.def("cross_entropy(Tensor p, Tensor q, int dim, str input_type, str reduction, float? base) -> Tensor");
  module.def("cross_entropy_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor)");
  module.def("cross_entropy_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base) -> (Tensor, Tensor, Tensor)");
  module.def("chi_squared_divergence(Tensor p, Tensor q, int dim, str reduction) -> Tensor");
  module.def("chi_squared_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str reduction) -> (Tensor, Tensor)");
  module.def("chi_squared_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str reduction) -> (Tensor, Tensor, Tensor)");

  // space_partitioning
  // Batched tree build - always use this, even for single trees (pass B=1)
  // Input: points (B, n, d), leaf_size
  // Returns: tuple of pre-padded (B, max_*) tensors for efficient consumption
  // (points, split_dim, split_val, left, right, indices, leaf_starts, leaf_counts)
  module.def("kd_tree_build_batched(Tensor points, int leaf_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // k-nearest neighbors query
  module.def("k_nearest_neighbors(Tensor points, Tensor split_dim, Tensor split_val, Tensor left, Tensor right, Tensor indices, Tensor leaf_starts, Tensor leaf_counts, Tensor queries, int k, float p) -> (Tensor, Tensor)");

  // range search query (returns nested tensors)
  module.def("range_search(Tensor points, Tensor split_dim, Tensor split_val, Tensor left, Tensor right, Tensor indices, Tensor leaf_starts, Tensor leaf_counts, Tensor queries, float radius, float p) -> (Tensor, Tensor)");

  // space_partitioning.bvh
  module.def("bvh_build(Tensor vertices, Tensor faces) -> Tensor");
  module.def("bvh_destroy(int scene_handle) -> ()");

  // geometry.ray_intersect
  module.def("bvh_ray_intersect(int scene_handle, Tensor origins, Tensor directions) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // geometry.closest_point
  module.def("bvh_closest_point(int scene_handle, Tensor query_points) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // geometry.ray_occluded
  module.def("bvh_ray_occluded(int scene_handle, Tensor origins, Tensor directions) -> Tensor");

  // geometry.transform
  module.def("reflect(Tensor direction, Tensor normal) -> Tensor");
  module.def("reflect_backward(Tensor grad_output, Tensor direction, Tensor normal) -> (Tensor, Tensor)");

  module.def("refract(Tensor direction, Tensor normal, Tensor eta) -> Tensor");
  module.def("refract_backward(Tensor grad_output, Tensor direction, Tensor normal, Tensor eta) -> (Tensor, Tensor, Tensor)");

  // Quaternion operations
  module.def("quaternion_multiply(Tensor q1, Tensor q2) -> Tensor");
  module.def("quaternion_multiply_backward(Tensor grad_output, Tensor q1, Tensor q2) -> (Tensor, Tensor)");

  module.def("quaternion_inverse(Tensor q) -> Tensor");
  module.def("quaternion_inverse_backward(Tensor grad_output, Tensor q) -> Tensor");

  module.def("quaternion_normalize(Tensor q) -> Tensor");
  module.def("quaternion_normalize_backward(Tensor grad_output, Tensor q) -> Tensor");

  module.def("quaternion_apply(Tensor q, Tensor point) -> Tensor");
  module.def("quaternion_apply_backward(Tensor grad_output, Tensor q, Tensor point) -> (Tensor, Tensor)");

  module.def("quaternion_to_matrix(Tensor q) -> Tensor");
  module.def("quaternion_to_matrix_backward(Tensor grad_output, Tensor q) -> Tensor");

  module.def("matrix_to_quaternion(Tensor matrix) -> Tensor");
  module.def("matrix_to_quaternion_backward(Tensor grad_output, Tensor matrix) -> Tensor");

  module.def("quaternion_slerp(Tensor q1, Tensor q2, Tensor t) -> Tensor");
  module.def("quaternion_slerp_backward(Tensor grad_output, Tensor q1, Tensor q2, Tensor t) -> (Tensor, Tensor, Tensor)");

  // geometry.convex_hull
  module.def("convex_hull(Tensor points) -> "
             "(Tensor vertices, Tensor simplices, Tensor neighbors, "
             "Tensor equations, Tensor area, Tensor volume, "
             "Tensor n_vertices, Tensor n_facets)");

  // encryption
  module.def("chacha20(Tensor key, Tensor nonce, int num_bytes, int counter=0) -> Tensor");
  module.def("sha256(Tensor data) -> Tensor");

  // Privacy operators
  module.def("gaussian_mechanism(Tensor x, Tensor noise, float sigma) -> Tensor");
  module.def("gaussian_mechanism_backward(Tensor grad_output) -> Tensor");

  module.def("laplace_mechanism(Tensor x, Tensor noise, float b) -> Tensor");
  module.def("laplace_mechanism_backward(Tensor grad_output) -> Tensor");

  // Probability - Normal distribution
  module.def("normal_cumulative_distribution(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_cumulative_distribution_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_cumulative_distribution_backward_backward(Tensor grad_grad_x, Tensor grad_grad_loc, Tensor grad_grad_scale, Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("normal_probability_density(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_probability_density_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_quantile(Tensor p, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_quantile_backward(Tensor grad, Tensor p, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_survival(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_survival_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_log_probability_density(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_log_probability_density_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");

  // Probability - Chi-squared distribution
  module.def("chi2_cumulative_distribution(Tensor x, Tensor df) -> Tensor");
  module.def("chi2_cumulative_distribution_backward(Tensor grad, Tensor x, Tensor df) -> (Tensor, Tensor)");
  module.def("chi2_probability_density(Tensor x, Tensor df) -> Tensor");
  module.def("chi2_probability_density_backward(Tensor grad, Tensor x, Tensor df) -> (Tensor, Tensor)");
  module.def("chi2_quantile(Tensor p, Tensor df) -> Tensor");
  module.def("chi2_quantile_backward(Tensor grad, Tensor p, Tensor df) -> (Tensor, Tensor)");
  module.def("chi2_survival(Tensor x, Tensor df) -> Tensor");
  module.def("chi2_survival_backward(Tensor grad, Tensor x, Tensor df) -> (Tensor, Tensor)");

  // Probability - F distribution
  module.def("f_cumulative_distribution(Tensor x, Tensor dfn, Tensor dfd) -> Tensor");
  module.def("f_cumulative_distribution_backward(Tensor grad, Tensor x, Tensor dfn, Tensor dfd) -> (Tensor, Tensor, Tensor)");
  module.def("f_probability_density(Tensor x, Tensor dfn, Tensor dfd) -> Tensor");
  module.def("f_probability_density_backward(Tensor grad, Tensor x, Tensor dfn, Tensor dfd) -> (Tensor, Tensor, Tensor)");
  module.def("f_quantile(Tensor p, Tensor dfn, Tensor dfd) -> Tensor");
  module.def("f_quantile_backward(Tensor grad, Tensor p, Tensor dfn, Tensor dfd) -> (Tensor, Tensor, Tensor)");
  module.def("f_survival(Tensor x, Tensor dfn, Tensor dfd) -> Tensor");
  module.def("f_survival_backward(Tensor grad, Tensor x, Tensor dfn, Tensor dfd) -> (Tensor, Tensor, Tensor)");

  // Probability - Beta distribution
  module.def("beta_cumulative_distribution(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("beta_cumulative_distribution_backward(Tensor grad, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("beta_probability_density(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("beta_probability_density_backward(Tensor grad, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("beta_quantile(Tensor p, Tensor a, Tensor b) -> Tensor");
  module.def("beta_quantile_backward(Tensor grad, Tensor p, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Probability - Gamma distribution
  module.def("gamma_cumulative_distribution(Tensor x, Tensor shape, Tensor scale) -> Tensor");
  module.def("gamma_cumulative_distribution_backward(Tensor grad, Tensor x, Tensor shape, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("gamma_probability_density(Tensor x, Tensor shape, Tensor scale) -> Tensor");
  module.def("gamma_probability_density_backward(Tensor grad, Tensor x, Tensor shape, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("gamma_quantile(Tensor p, Tensor shape, Tensor scale) -> Tensor");
  module.def("gamma_quantile_backward(Tensor grad, Tensor p, Tensor shape, Tensor scale) -> (Tensor, Tensor, Tensor)");

  // Probability - Binomial distribution
  module.def("binomial_cumulative_distribution(Tensor k, Tensor n, Tensor p) -> Tensor");
  module.def("binomial_cumulative_distribution_backward(Tensor grad, Tensor k, Tensor n, Tensor p) -> (Tensor, Tensor, Tensor)");
  module.def("binomial_probability_mass(Tensor k, Tensor n, Tensor p) -> Tensor");
  module.def("binomial_probability_mass_backward(Tensor grad, Tensor k, Tensor n, Tensor p) -> (Tensor, Tensor, Tensor)");

  // Probability - Poisson distribution
  module.def("poisson_cumulative_distribution(Tensor k, Tensor rate) -> Tensor");
  module.def("poisson_cumulative_distribution_backward(Tensor grad, Tensor k, Tensor rate) -> (Tensor, Tensor)");
  module.def("poisson_probability_mass(Tensor k, Tensor rate) -> Tensor");
  module.def("poisson_probability_mass_backward(Tensor grad, Tensor k, Tensor rate) -> (Tensor, Tensor)");

  // coding - Morton encoding (Z-order curve)
  module.def("morton_encode(Tensor coordinates) -> Tensor");
  module.def("morton_decode(Tensor codes, int dimensions) -> Tensor");

  // space_partitioning - Octree
  // Construction (returns: codes, data, structure, children_mask, weights, maximum_depth, count)
  // aggregation: int (0=mean, 1=sum, 2=max)
  module.def("octree_build(Tensor points, Tensor data, int maximum_depth, float capacity_factor, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Point queries (with query_depth and found mask)
  // interpolation: int (0=nearest, 1=trilinear)
  module.def("octree_sample(Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor points, int maximum_depth, int interpolation, int? query_depth) -> (Tensor, Tensor)");

  // Backward for octree_sample
  // Returns: (grad_data, grad_points) - grad_points is zeros for nearest interpolation
  module.def("octree_sample_backward(Tensor grad_output, Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor points, int maximum_depth, int interpolation, int? query_depth) -> (Tensor, Tensor)");

  // Ray marching (hierarchical DDA traversal)
  // Returns: (positions, data, mask) with fixed maximum_steps size
  module.def("octree_ray_marching(Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor origins, Tensor directions, int maximum_depth, float? step_size, int maximum_steps) -> (Tensor, Tensor, Tensor)");

  // Backward for octree_ray_marching
  // Returns: (grad_data, grad_origins, grad_directions)
  // grad_origins and grad_directions are zeros for adaptive stepping mode
  module.def("octree_ray_marching_backward(Tensor grad_positions, Tensor grad_data_out, Tensor mask, Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor origins, Tensor directions, int maximum_depth, float? step_size, int maximum_steps) -> (Tensor, Tensor, Tensor)");

  // Neighbor finding (LOD-aware)
  // connectivity: 6 (face), 18 (face+edge), 26 (face+edge+corner)
  // Returns: (neighbor_codes, neighbor_data) where neighbor_codes[i] is -1 if no neighbor exists
  module.def("octree_neighbors(Tensor data, Tensor codes, Tensor structure, Tensor children_mask, Tensor query_codes, int connectivity) -> (Tensor, Tensor)");

  // Dynamic Updates (Phase 5)
  // All dynamic update operations rebuild the hash table and return the same 7-tuple as octree_build:
  // (codes, data, structure, children_mask, weights, hash_table, hash_table_offsets)
  // aggregation: 0=mean, 1=sum, 2=max

  // Insert new voxels at specified depth, auto-creates ancestors with aggregated data
  module.def("octree_insert(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor new_points, Tensor new_data, int depth, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Remove voxels by code, prunes empty ancestors
  module.def("octree_remove(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor remove_codes, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Subdivide leaf voxel into 8 children, distributes parent data to children
  module.def("octree_subdivide(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor subdivide_codes, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Merge 8 sibling leaves into parent, aggregates child data
  module.def("octree_merge(Tensor codes, Tensor data, Tensor structure, Tensor children_mask, Tensor weights, Tensor merge_codes, int maximum_depth, int aggregation) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // polynomial evaluation operators
  // Forward: coeffs (B, N) x points (M,) -> output (B, M)
  module.def("polynomial_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("polynomial_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("polynomial_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  module.def("polynomial_derivative(Tensor coeffs) -> Tensor");
  module.def("polynomial_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("polynomial_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  module.def("polynomial_antiderivative(Tensor coeffs, Tensor constant) -> Tensor");
  module.def("polynomial_antiderivative_backward(Tensor grad_output, Tensor coeffs, Tensor constant) -> (Tensor, Tensor)");
  module.def("polynomial_antiderivative_backward_backward(Tensor gg_coeffs, Tensor gg_constant, Tensor coeffs) -> Tensor");

  // Polynomial coefficient-wise operations
  module.def("polynomial_add(Tensor p, Tensor q) -> Tensor");
  module.def("polynomial_add_backward(Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor)");
  module.def("polynomial_add_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor)");

  module.def("polynomial_subtract(Tensor p, Tensor q) -> Tensor");
  module.def("polynomial_subtract_backward(Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor)");
  module.def("polynomial_subtract_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor)");

  module.def("polynomial_negate(Tensor p) -> Tensor");
  module.def("polynomial_negate_backward(Tensor grad_output, Tensor p) -> Tensor");
  module.def("polynomial_negate_backward_backward(Tensor gg_p, Tensor grad_output, Tensor p) -> (Tensor, Tensor)");

  module.def("polynomial_scale(Tensor p, Tensor c) -> Tensor");
  module.def("polynomial_scale_backward(Tensor grad_output, Tensor p, Tensor c) -> (Tensor, Tensor)");
  module.def("polynomial_scale_backward_backward(Tensor gg_p, Tensor gg_c, Tensor grad_output, Tensor p, Tensor c) -> (Tensor, Tensor, Tensor)");

  // polynomial multiplication (discrete convolution)
  // Forward: p (B, N), q (B, M) -> output (B, N+M-1)
  module.def("polynomial_multiply(Tensor p, Tensor q) -> Tensor");
  module.def("polynomial_multiply_backward(Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor)");
  module.def("polynomial_multiply_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor)");

  // polynomial division with remainder (long division)
  // Forward: p (B, N), q (B, M) -> (quotient (B, N-M+1), remainder (B, max(M-1, 1)))
  module.def("polynomial_divmod(Tensor p, Tensor q) -> (Tensor, Tensor)");
  module.def("polynomial_divmod_backward(Tensor grad_Q, Tensor grad_R, Tensor Q, Tensor p, Tensor q) -> (Tensor, Tensor)");
  module.def("polynomial_divmod_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_Q, Tensor grad_R, Tensor Q, Tensor p, Tensor q) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Chebyshev T polynomial evaluation (Clenshaw's algorithm)
  module.def("chebyshev_polynomial_t_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("chebyshev_polynomial_t_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_t_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Chebyshev T polynomial multiplication (linearization formula)
  module.def("chebyshev_polynomial_t_multiply(Tensor a, Tensor b) -> Tensor");
  module.def("chebyshev_polynomial_t_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_t_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Chebyshev T polynomial derivative (recurrence formula)
  module.def("chebyshev_polynomial_t_derivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_t_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_t_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Chebyshev T polynomial multiply by x (shift operation)
  module.def("chebyshev_polynomial_t_mulx(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_t_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_t_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  // Chebyshev T polynomial antiderivative
  module.def("chebyshev_polynomial_t_antiderivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_t_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_t_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Chebyshev U polynomial evaluation (Clenshaw's algorithm)
  module.def("chebyshev_polynomial_u_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("chebyshev_polynomial_u_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_u_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Chebyshev U polynomial multiplication (linearization formula)
  module.def("chebyshev_polynomial_u_multiply(Tensor a, Tensor b) -> Tensor");
  module.def("chebyshev_polynomial_u_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_u_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Chebyshev U polynomial derivative (recurrence formula)
  module.def("chebyshev_polynomial_u_derivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_u_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_u_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Chebyshev U polynomial multiply by x (shift operation)
  module.def("chebyshev_polynomial_u_mulx(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_u_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_u_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  // Chebyshev U polynomial antiderivative
  module.def("chebyshev_polynomial_u_antiderivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_u_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_u_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Chebyshev V polynomial evaluation (Clenshaw's algorithm)
  module.def("chebyshev_polynomial_v_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("chebyshev_polynomial_v_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_v_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Chebyshev V polynomial multiplication (linearization formula)
  module.def("chebyshev_polynomial_v_multiply(Tensor a, Tensor b) -> Tensor");
  module.def("chebyshev_polynomial_v_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_v_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Chebyshev V polynomial derivative (recurrence formula)
  module.def("chebyshev_polynomial_v_derivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_v_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_v_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Chebyshev V polynomial multiply by x (shift operation)
  module.def("chebyshev_polynomial_v_mulx(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_v_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_v_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  // Chebyshev V polynomial antiderivative
  module.def("chebyshev_polynomial_v_antiderivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_v_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_v_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Chebyshev W polynomial evaluation (Clenshaw's algorithm)
  module.def("chebyshev_polynomial_w_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("chebyshev_polynomial_w_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_w_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Chebyshev W polynomial multiplication (linearization formula)
  module.def("chebyshev_polynomial_w_multiply(Tensor a, Tensor b) -> Tensor");
  module.def("chebyshev_polynomial_w_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_w_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Chebyshev W polynomial derivative (recurrence formula)
  module.def("chebyshev_polynomial_w_derivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_w_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_w_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Chebyshev W polynomial multiply by x (shift operation)
  module.def("chebyshev_polynomial_w_mulx(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_w_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_w_mulx_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> (Tensor, Tensor)");

  // Chebyshev W polynomial antiderivative
  module.def("chebyshev_polynomial_w_antiderivative(Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_w_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("chebyshev_polynomial_w_antiderivative_backward_backward(Tensor gg_coeffs, Tensor grad_output, Tensor coeffs) -> Tensor");

  // Legendre P polynomial evaluation (Clenshaw's algorithm)
  module.def("legendre_polynomial_p_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("legendre_polynomial_p_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("legendre_polynomial_p_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Legendre P polynomial derivative
  module.def("legendre_polynomial_p_derivative(Tensor coeffs) -> Tensor");
  module.def("legendre_polynomial_p_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("legendre_polynomial_p_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Legendre P polynomial antiderivative
  module.def("legendre_polynomial_p_antiderivative(Tensor coeffs) -> Tensor");
  module.def("legendre_polynomial_p_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("legendre_polynomial_p_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Legendre P polynomial mulx
  module.def("legendre_polynomial_p_mulx(Tensor coeffs) -> Tensor");
  module.def("legendre_polynomial_p_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("legendre_polynomial_p_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Legendre P polynomial multiply
  module.def("legendre_polynomial_p_multiply(Tensor a, Tensor b) -> Tensor");
  module.def("legendre_polynomial_p_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("legendre_polynomial_p_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Laguerre L polynomial evaluation (Clenshaw's algorithm)
  module.def("laguerre_polynomial_l_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("laguerre_polynomial_l_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("laguerre_polynomial_l_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Laguerre L polynomial derivative
  module.def("laguerre_polynomial_l_derivative(Tensor coeffs) -> Tensor");
  module.def("laguerre_polynomial_l_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("laguerre_polynomial_l_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Laguerre L polynomial antiderivative
  module.def("laguerre_polynomial_l_antiderivative(Tensor coeffs) -> Tensor");
  module.def("laguerre_polynomial_l_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("laguerre_polynomial_l_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Laguerre L polynomial mulx
  module.def("laguerre_polynomial_l_mulx(Tensor coeffs) -> Tensor");
  module.def("laguerre_polynomial_l_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("laguerre_polynomial_l_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Laguerre L polynomial multiply
  module.def("laguerre_polynomial_l_multiply(Tensor a, Tensor b) -> Tensor");
  module.def("laguerre_polynomial_l_multiply_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("laguerre_polynomial_l_multiply_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  // Hermite H (physicists') polynomial evaluation (Clenshaw's algorithm)
  module.def("hermite_polynomial_h_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("hermite_polynomial_h_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("hermite_polynomial_h_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Hermite H polynomial derivative
  module.def("hermite_polynomial_h_derivative(Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_h_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_h_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Hermite H polynomial antiderivative
  module.def("hermite_polynomial_h_antiderivative(Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_h_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_h_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Hermite H polynomial mulx (multiply by x)
  module.def("hermite_polynomial_h_mulx(Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_h_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_h_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Hermite He (probabilists') polynomial evaluation (Clenshaw's algorithm)
  module.def("hermite_polynomial_he_evaluate(Tensor coeffs, Tensor x) -> Tensor");
  module.def("hermite_polynomial_he_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor)");
  module.def("hermite_polynomial_he_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor grad_output, Tensor coeffs, Tensor x) -> (Tensor, Tensor, Tensor)");

  // Hermite He polynomial derivative
  module.def("hermite_polynomial_he_derivative(Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_he_derivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_he_derivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Hermite He polynomial antiderivative
  module.def("hermite_polynomial_he_antiderivative(Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_he_antiderivative_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_he_antiderivative_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Hermite He polynomial mulx (multiply by x)
  module.def("hermite_polynomial_he_mulx(Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_he_mulx_backward(Tensor grad_output, Tensor coeffs) -> Tensor");
  module.def("hermite_polynomial_he_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs) -> Tensor");

  // Gegenbauer C (ultraspherical) polynomial evaluation (Clenshaw's algorithm)
  module.def("gegenbauer_polynomial_c_evaluate(Tensor coeffs, Tensor x, Tensor alpha) -> Tensor");
  module.def("gegenbauer_polynomial_c_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha) -> (Tensor, Tensor, Tensor)");
  module.def("gegenbauer_polynomial_c_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor gg_alpha, Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha) -> (Tensor, Tensor, Tensor, Tensor)");

  // Gegenbauer C polynomial mulx (multiply by x)
  module.def("gegenbauer_polynomial_c_mulx(Tensor coeffs, Tensor alpha) -> Tensor");
  module.def("gegenbauer_polynomial_c_mulx_backward(Tensor grad_output, Tensor coeffs, Tensor alpha) -> (Tensor, Tensor)");
  module.def("gegenbauer_polynomial_c_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs, Tensor alpha) -> Tensor");

  // Jacobi P polynomial evaluation (forward recurrence)
  module.def("jacobi_polynomial_p_evaluate(Tensor coeffs, Tensor x, Tensor alpha, Tensor beta) -> Tensor");
  module.def("jacobi_polynomial_p_evaluate_backward(Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha, Tensor beta) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("jacobi_polynomial_p_evaluate_backward_backward(Tensor gg_coeffs, Tensor gg_x, Tensor gg_alpha, Tensor gg_beta, Tensor grad_output, Tensor coeffs, Tensor x, Tensor alpha, Tensor beta) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Jacobi P polynomial mulx (multiply by x)
  module.def("jacobi_polynomial_p_mulx(Tensor coeffs, Tensor alpha, Tensor beta) -> Tensor");
  module.def("jacobi_polynomial_p_mulx_backward(Tensor grad_output, Tensor coeffs, Tensor alpha, Tensor beta) -> (Tensor, Tensor, Tensor)");
  module.def("jacobi_polynomial_p_mulx_backward_backward(Tensor gg_coeffs, Tensor coeffs, Tensor alpha, Tensor beta) -> Tensor");

  // linear_algebra decomposition
  module.def("symmetric_generalized_eigenvalue(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("generalized_eigenvalue(Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("schur_decomposition(Tensor a, str output='real') -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("polar_decomposition(Tensor a, str side='right') -> (Tensor, Tensor, Tensor)");
  module.def("hessenberg(Tensor a) -> (Tensor, Tensor, Tensor)");
  module.def("generalized_schur(Tensor a, Tensor b, str output='real') -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def("jordan_decomposition(Tensor a) -> (Tensor, Tensor, Tensor)");
  module.def("pivoted_lu(Tensor a) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("pivoted_qr(Tensor a) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("rank_revealing_qr(Tensor a, float tol=1e-10) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def("ldl_decomposition(Tensor a) -> (Tensor, Tensor, Tensor, Tensor)");

  // morphology
  module.def("erosion(Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");
  module.def("erosion_backward(Tensor grad_output, Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");

  module.def("dilation(Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");
  module.def("dilation_backward(Tensor grad_output, Tensor input, Tensor structuring_element, int[]? origin, int padding_mode) -> Tensor");

  // pad
  module.def("pad(Tensor input, int[] padding, str mode, float value, int[]? dim, int order, Tensor? out) -> Tensor");
  module.def("pad_backward(Tensor grad_output, int[] input_shape, int[] padding, str mode, int[]? dim, int order) -> Tensor");
  module.def("pad_backward_backward(Tensor grad_grad_input, int[] padding, str mode, int[]? dim, int order) -> Tensor");
}
