#pragma once

#include "macros.h"

#include "../kernel/special_functions/gamma.h"
#include "../kernel/special_functions/gamma_backward.h"
#include "../kernel/special_functions/gamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(gamma, z)

#include "../kernel/special_functions/digamma.h"
#include "../kernel/special_functions/digamma_backward.h"
#include "../kernel/special_functions/digamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(digamma, z)

#include "../kernel/special_functions/trigamma.h"
#include "../kernel/special_functions/trigamma_backward.h"
#include "../kernel/special_functions/trigamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(trigamma, z)

#include "../kernel/special_functions/beta.h"
#include "../kernel/special_functions/beta_backward.h"
#include "../kernel/special_functions/beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(beta, a, b)

#include "../kernel/special_functions/chebyshev_polynomial_t.h"
#include "../kernel/special_functions/chebyshev_polynomial_t_backward.h"
#include "../kernel/special_functions/chebyshev_polynomial_t_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_t, x, n)

#include "../kernel/special_functions/incomplete_beta.h"
#include "../kernel/special_functions/incomplete_beta_backward.h"
#include "../kernel/special_functions/incomplete_beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(incomplete_beta, x, a, b)

#include "../kernel/special_functions/hypergeometric_2_f_1.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(hypergeometric_2_f_1, a, b, c, z)

#include "../kernel/special_functions/polygamma.h"
#include "../kernel/special_functions/polygamma_backward.h"
#include "../kernel/special_functions/polygamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(polygamma, n, z)

#include "../kernel/special_functions/log_beta.h"
#include "../kernel/special_functions/log_beta_backward.h"
#include "../kernel/special_functions/log_beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(log_beta, a, b)

#include "../kernel/special_functions/log_gamma.h"
#include "../kernel/special_functions/log_gamma_backward.h"
#include "../kernel/special_functions/log_gamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(log_gamma, z)

#include "../kernel/special_functions/regularized_gamma_p.h"
#include "../kernel/special_functions/regularized_gamma_p_backward.h"
#include "../kernel/special_functions/regularized_gamma_p_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(regularized_gamma_p, a, x)

#include "../kernel/special_functions/regularized_gamma_q.h"
#include "../kernel/special_functions/regularized_gamma_q_backward.h"
#include "../kernel/special_functions/regularized_gamma_q_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(regularized_gamma_q, a, x)

#include "../kernel/special_functions/modified_bessel_i_0.h"
#include "../kernel/special_functions/modified_bessel_i_0_backward.h"
#include "../kernel/special_functions/modified_bessel_i_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(modified_bessel_i_0, z)

#include "../kernel/special_functions/modified_bessel_i_1.h"
#include "../kernel/special_functions/modified_bessel_i_1_backward.h"
#include "../kernel/special_functions/modified_bessel_i_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(modified_bessel_i_1, z)

#include "../kernel/special_functions/bessel_j_0.h"
#include "../kernel/special_functions/bessel_j_0_backward.h"
#include "../kernel/special_functions/bessel_j_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(bessel_j_0, z)

#include "../kernel/special_functions/bessel_j_1.h"
#include "../kernel/special_functions/bessel_j_1_backward.h"
#include "../kernel/special_functions/bessel_j_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(bessel_j_1, z)

#include "../kernel/special_functions/bessel_y_0.h"
#include "../kernel/special_functions/bessel_y_0_backward.h"
#include "../kernel/special_functions/bessel_y_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(bessel_y_0, z)

#include "../kernel/special_functions/bessel_y_1.h"
#include "../kernel/special_functions/bessel_y_1_backward.h"
#include "../kernel/special_functions/bessel_y_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(bessel_y_1, z)

#include "../kernel/special_functions/modified_bessel_k_0.h"
#include "../kernel/special_functions/modified_bessel_k_0_backward.h"
#include "../kernel/special_functions/modified_bessel_k_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(modified_bessel_k_0, z)

#include "../kernel/special_functions/modified_bessel_k_1.h"
#include "../kernel/special_functions/modified_bessel_k_1_backward.h"
#include "../kernel/special_functions/modified_bessel_k_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(modified_bessel_k_1, z)

#include "../kernel/special_functions/bessel_j.h"
#include "../kernel/special_functions/bessel_j_backward.h"
#include "../kernel/special_functions/bessel_j_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(bessel_j, n, z)

#include "../kernel/special_functions/bessel_y.h"
#include "../kernel/special_functions/bessel_y_backward.h"
#include "../kernel/special_functions/bessel_y_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(bessel_y, n, z)

#include "../kernel/special_functions/modified_bessel_k.h"
#include "../kernel/special_functions/modified_bessel_k_backward.h"
#include "../kernel/special_functions/modified_bessel_k_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(modified_bessel_k, n, z)

#include "../kernel/special_functions/modified_bessel_i.h"
#include "../kernel/special_functions/modified_bessel_i_backward.h"
#include "../kernel/special_functions/modified_bessel_i_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(modified_bessel_i, n, z)
#include "../kernel/special_functions/carlson_elliptic_integral_r_f.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_f_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_f_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_f, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_d.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_d_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_d_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_d, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_c.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_c_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_c_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_c, x, y)

#include "../kernel/special_functions/carlson_elliptic_integral_r_j.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_j_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_j_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_j, x, y, z, p)

#include "../kernel/special_functions/carlson_elliptic_integral_r_g.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_g_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_g_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_g, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_e.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_e_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_e_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_e, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_m.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_m_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_m_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_m, x, y, z)

#include "../kernel/special_functions/carlson_elliptic_integral_r_k.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_k_backward.h"
#include "../kernel/special_functions/carlson_elliptic_integral_r_k_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(carlson_elliptic_integral_r_k, x, y)

#include "../kernel/special_functions/complete_legendre_elliptic_integral_k.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_k_backward.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_k_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(complete_legendre_elliptic_integral_k, m)

#include "../kernel/special_functions/complete_legendre_elliptic_integral_e.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_e_backward.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_e_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(complete_legendre_elliptic_integral_e, m)

#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_e.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_e_backward.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_e_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(incomplete_legendre_elliptic_integral_e, phi, m)

#include "../kernel/special_functions/complete_legendre_elliptic_integral_pi.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_pi_backward.h"
#include "../kernel/special_functions/complete_legendre_elliptic_integral_pi_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(complete_legendre_elliptic_integral_pi, n, m)

#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_pi.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_pi_backward.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_pi_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(incomplete_legendre_elliptic_integral_pi, n, phi, m)

#include "../kernel/special_functions/jacobi_amplitude_am.h"
#include "../kernel/special_functions/jacobi_amplitude_am_backward.h"
#include "../kernel/special_functions/jacobi_amplitude_am_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_amplitude_am, u, m)

#include "../kernel/special_functions/jacobi_elliptic_dn.h"
#include "../kernel/special_functions/jacobi_elliptic_dn_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_dn_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_dn, u, m)

#include "../kernel/special_functions/jacobi_elliptic_cn.h"
#include "../kernel/special_functions/jacobi_elliptic_cn_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_cn_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_cn, u, m)

#include "../kernel/special_functions/jacobi_elliptic_sn.h"
#include "../kernel/special_functions/jacobi_elliptic_sn_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_sn_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_sn, u, m)

#include "../kernel/special_functions/jacobi_elliptic_sd.h"
#include "../kernel/special_functions/jacobi_elliptic_sd_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_sd_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_sd, u, m)

#include "../kernel/special_functions/jacobi_elliptic_cd.h"
#include "../kernel/special_functions/jacobi_elliptic_cd_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_cd_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_cd, u, m)

#include "../kernel/special_functions/jacobi_elliptic_sc.h"
#include "../kernel/special_functions/jacobi_elliptic_sc_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_sc_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_sc, u, m)

#include "../kernel/special_functions/jacobi_elliptic_nd.h"
#include "../kernel/special_functions/jacobi_elliptic_nd_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_nd_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_nd, u, m)

#include "../kernel/special_functions/jacobi_elliptic_nc.h"
#include "../kernel/special_functions/jacobi_elliptic_nc_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_nc_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_nc, u, m)

#include "../kernel/special_functions/jacobi_elliptic_ns.h"
#include "../kernel/special_functions/jacobi_elliptic_ns_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_ns_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_ns, u, m)

#include "../kernel/special_functions/jacobi_elliptic_dc.h"
#include "../kernel/special_functions/jacobi_elliptic_dc_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_dc_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_dc, u, m)

#include "../kernel/special_functions/jacobi_elliptic_ds.h"
#include "../kernel/special_functions/jacobi_elliptic_ds_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_ds_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_ds, u, m)

#include "../kernel/special_functions/jacobi_elliptic_cs.h"
#include "../kernel/special_functions/jacobi_elliptic_cs_backward.h"
#include "../kernel/special_functions/jacobi_elliptic_cs_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(jacobi_elliptic_cs, u, m)

// Inverse Jacobi elliptic functions (primary: sn, cn, dn)
#include "../kernel/special_functions/inverse_jacobi_elliptic_sn.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sn_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sn_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(inverse_jacobi_elliptic_sn, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_cn.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cn_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cn_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(inverse_jacobi_elliptic_cn, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_dn.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_dn_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_dn_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(inverse_jacobi_elliptic_dn, x, m)

// Inverse Jacobi elliptic functions (derived: sd, cd, sc)
#include "../kernel/special_functions/inverse_jacobi_elliptic_sd.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sd_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sd_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(inverse_jacobi_elliptic_sd, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_cd.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cd_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_cd_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(inverse_jacobi_elliptic_cd, x, m)

#include "../kernel/special_functions/inverse_jacobi_elliptic_sc.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sc_backward.h"
#include "../kernel/special_functions/inverse_jacobi_elliptic_sc_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(inverse_jacobi_elliptic_sc, x, m)

// Jacobi theta functions
#include "../kernel/special_functions/theta_1.h"
#include "../kernel/special_functions/theta_1_backward.h"
#include "../kernel/special_functions/theta_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(theta_1, z, q)

#include "../kernel/special_functions/theta_2.h"
#include "../kernel/special_functions/theta_2_backward.h"
#include "../kernel/special_functions/theta_2_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(theta_2, z, q)

#include "../kernel/special_functions/theta_3.h"
#include "../kernel/special_functions/theta_3_backward.h"
#include "../kernel/special_functions/theta_3_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(theta_3, z, q)

#include "../kernel/special_functions/theta_4.h"
#include "../kernel/special_functions/theta_4_backward.h"
#include "../kernel/special_functions/theta_4_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(theta_4, z, q)

#include "../kernel/special_functions/spherical_bessel_j_0.h"
#include "../kernel/special_functions/spherical_bessel_j_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_j_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_j_0, z)

#include "../kernel/special_functions/spherical_bessel_j_1.h"
#include "../kernel/special_functions/spherical_bessel_j_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_j_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_j_1, z)

#include "../kernel/special_functions/spherical_bessel_j.h"
#include "../kernel/special_functions/spherical_bessel_j_backward.h"
#include "../kernel/special_functions/spherical_bessel_j_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(spherical_bessel_j, n, z)

#include "../kernel/special_functions/spherical_bessel_y_0.h"
#include "../kernel/special_functions/spherical_bessel_y_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_y_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_y_0, z)

#include "../kernel/special_functions/spherical_bessel_y_1.h"
#include "../kernel/special_functions/spherical_bessel_y_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_y_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_y_1, z)

#include "../kernel/special_functions/spherical_bessel_y.h"
#include "../kernel/special_functions/spherical_bessel_y_backward.h"
#include "../kernel/special_functions/spherical_bessel_y_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(spherical_bessel_y, n, z)

#include "../kernel/special_functions/spherical_bessel_i_0.h"
#include "../kernel/special_functions/spherical_bessel_i_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_i_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_i_0, z)

#include "../kernel/special_functions/spherical_bessel_i_1.h"
#include "../kernel/special_functions/spherical_bessel_i_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_i_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_i_1, z)

#include "../kernel/special_functions/spherical_bessel_i.h"
#include "../kernel/special_functions/spherical_bessel_i_backward.h"
#include "../kernel/special_functions/spherical_bessel_i_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(spherical_bessel_i, n, z)

#include "../kernel/special_functions/spherical_bessel_k_0.h"
#include "../kernel/special_functions/spherical_bessel_k_0_backward.h"
#include "../kernel/special_functions/spherical_bessel_k_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_k_0, z)

#include "../kernel/special_functions/spherical_bessel_k_1.h"
#include "../kernel/special_functions/spherical_bessel_k_1_backward.h"
#include "../kernel/special_functions/spherical_bessel_k_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(spherical_bessel_k_1, z)

#include "../kernel/special_functions/spherical_bessel_k.h"
#include "../kernel/special_functions/spherical_bessel_k_backward.h"
#include "../kernel/special_functions/spherical_bessel_k_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(spherical_bessel_k, n, z)

#include "../kernel/special_functions/exponential_integral_ei.h"
#include "../kernel/special_functions/exponential_integral_ei_backward.h"
#include "../kernel/special_functions/exponential_integral_ei_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(exponential_integral_ei, x)

#include "../kernel/special_functions/exponential_integral_e_1.h"
#include "../kernel/special_functions/exponential_integral_e_1_backward.h"
#include "../kernel/special_functions/exponential_integral_e_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(exponential_integral_e_1, x)

#include "../kernel/special_functions/exponential_integral_ein.h"
#include "../kernel/special_functions/exponential_integral_ein_backward.h"
#include "../kernel/special_functions/exponential_integral_ein_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(exponential_integral_ein, x)

#include "../kernel/special_functions/exponential_integral_e.h"
#include "../kernel/special_functions/exponential_integral_e_backward.h"
#include "../kernel/special_functions/exponential_integral_e_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(exponential_integral_e, n, x)

#include "../kernel/special_functions/sine_integral_si.h"
#include "../kernel/special_functions/sine_integral_si_backward.h"
#include "../kernel/special_functions/sine_integral_si_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(sine_integral_si, x)

#include "../kernel/special_functions/cosine_integral_ci.h"
#include "../kernel/special_functions/cosine_integral_ci_backward.h"
#include "../kernel/special_functions/cosine_integral_ci_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(cosine_integral_ci, x)
