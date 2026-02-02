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

#include "../kernel/special_functions/chebyshev_polynomial_u.h"
#include "../kernel/special_functions/chebyshev_polynomial_u_backward.h"
#include "../kernel/special_functions/chebyshev_polynomial_u_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_u, x, n)

#include "../kernel/special_functions/incomplete_beta.h"
#include "../kernel/special_functions/incomplete_beta_backward.h"
#include "../kernel/special_functions/incomplete_beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(incomplete_beta, x, a, b)

#include "../kernel/special_functions/hypergeometric_2_f_1.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward.h"
#include "../kernel/special_functions/hypergeometric_2_f_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(hypergeometric_2_f_1, a, b, c, z)

#include "../kernel/special_functions/confluent_hypergeometric_m.h"
#include "../kernel/special_functions/confluent_hypergeometric_m_backward.h"
#include "../kernel/special_functions/confluent_hypergeometric_m_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(confluent_hypergeometric_m, a, b, z)

#include "../kernel/special_functions/confluent_hypergeometric_u.h"
#include "../kernel/special_functions/confluent_hypergeometric_u_backward.h"
#include "../kernel/special_functions/confluent_hypergeometric_u_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(confluent_hypergeometric_u, a, b, z)

#include "../kernel/special_functions/whittaker_m.h"
#include "../kernel/special_functions/whittaker_m_backward.h"
#include "../kernel/special_functions/whittaker_m_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(whittaker_m, kappa, mu, z)

#include "../kernel/special_functions/whittaker_w.h"
#include "../kernel/special_functions/whittaker_w_backward.h"
#include "../kernel/special_functions/whittaker_w_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(whittaker_w, kappa, mu, z)

#include "../kernel/special_functions/hypergeometric_0_f_1.h"
#include "../kernel/special_functions/hypergeometric_0_f_1_backward.h"
#include "../kernel/special_functions/hypergeometric_0_f_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(hypergeometric_0_f_1, b, z)

#include "../kernel/special_functions/hypergeometric_1_f_2.h"
#include "../kernel/special_functions/hypergeometric_1_f_2_backward.h"
#include "../kernel/special_functions/hypergeometric_1_f_2_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(hypergeometric_1_f_2, a, b1, b2, z)

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

#include "../kernel/special_functions/reciprocal_gamma.h"
#include "../kernel/special_functions/reciprocal_gamma_backward.h"
#include "../kernel/special_functions/reciprocal_gamma_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(reciprocal_gamma, z)

#include "../kernel/special_functions/gamma_sign.h"
#include "../kernel/special_functions/gamma_sign_backward.h"
#include "../kernel/special_functions/gamma_sign_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR(gamma_sign, x)

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

#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_f.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_f_backward.h"
#include "../kernel/special_functions/incomplete_legendre_elliptic_integral_f_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(incomplete_legendre_elliptic_integral_f, phi, m)

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

// Weierstrass P function
// Custom implementation because weierstrass_p uses complex operations
// not supported by Half/BFloat16
#include "../kernel/special_functions/weierstrass_p.h"
#include "../kernel/special_functions/weierstrass_p_backward.h"
#include "../kernel/special_functions/weierstrass_p_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor weierstrass_p(
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor output;
    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_p",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_p(z, g2, g3);
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_p",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_p(z, g2, g3);
                    }
                );
            }
        );
    }

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_p_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor gradient_z, gradient_g2, gradient_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_z)
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_input(gradient_input)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_p_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_p_backward(
                            gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_p_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_p_backward(
                            gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1),
        iterator.output(2)
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> weierstrass_p_backward_backward(
    const at::Tensor &gg_z_input,
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor grad_grad, grad_z, grad_g2, grad_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_z)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_input(gg_z_input)
        .add_input(gg_g2_input)
        .add_input(gg_g3_input)
        .add_input(gradient_input)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_p_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_z,
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t z,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_p_backward_backward(
                            gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_p_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_z,
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t z,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_p_backward_backward(
                            gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1),
        iterator.output(2),
        iterator.output(3)
    );
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("weierstrass_p", torchscience::cpu::special_functions::weierstrass_p);
    module.impl("weierstrass_p_backward", torchscience::cpu::special_functions::weierstrass_p_backward);
    module.impl("weierstrass_p_backward_backward", torchscience::cpu::special_functions::weierstrass_p_backward_backward);
}

// Weierstrass Sigma function
// Custom implementation because weierstrass_sigma uses complex operations
// not supported by Half/BFloat16
#include "../kernel/special_functions/weierstrass_sigma.h"
#include "../kernel/special_functions/weierstrass_sigma_backward.h"
#include "../kernel/special_functions/weierstrass_sigma_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor weierstrass_sigma(
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor output;
    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_sigma",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_sigma(z, g2, g3);
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_sigma",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_sigma(z, g2, g3);
                    }
                );
            }
        );
    }

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_sigma_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor gradient_z, gradient_g2, gradient_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_z)
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_input(gradient_input)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_sigma_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_sigma_backward(
                            gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_sigma_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_sigma_backward(
                            gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1),
        iterator.output(2)
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> weierstrass_sigma_backward_backward(
    const at::Tensor &gg_z_input,
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor grad_grad, grad_z, grad_g2, grad_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_z)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_input(gg_z_input)
        .add_input(gg_g2_input)
        .add_input(gg_g3_input)
        .add_input(gradient_input)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_sigma_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_z,
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t z,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_sigma_backward_backward(
                            gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_sigma_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_z,
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t z,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_sigma_backward_backward(
                            gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1),
        iterator.output(2),
        iterator.output(3)
    );
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("weierstrass_sigma", torchscience::cpu::special_functions::weierstrass_sigma);
    module.impl("weierstrass_sigma_backward", torchscience::cpu::special_functions::weierstrass_sigma_backward);
    module.impl("weierstrass_sigma_backward_backward", torchscience::cpu::special_functions::weierstrass_sigma_backward_backward);
}

// Weierstrass Zeta function
// Custom implementation because weierstrass_zeta uses complex operations
// not supported by Half/BFloat16
#include "../kernel/special_functions/weierstrass_zeta.h"
#include "../kernel/special_functions/weierstrass_zeta_backward.h"
#include "../kernel/special_functions/weierstrass_zeta_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor weierstrass_zeta(
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor output;
    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_zeta",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_zeta(z, g2, g3);
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_zeta",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z, scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_zeta(z, g2, g3);
                    }
                );
            }
        );
    }

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_zeta_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor gradient_z, gradient_g2, gradient_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_z)
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_input(gradient_input)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_zeta_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_zeta_backward(
                            gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_zeta_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t z, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_zeta_backward(
                            gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1),
        iterator.output(2)
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> weierstrass_zeta_backward_backward(
    const at::Tensor &gg_z_input,
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor grad_grad, grad_z, grad_g2, grad_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_z)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_input(gg_z_input)
        .add_input(gg_g2_input)
        .add_input(gg_g3_input)
        .add_input(gradient_input)
        .add_input(z_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_zeta_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_z,
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t z,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_zeta_backward_backward(
                            gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_zeta_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_z,
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t z,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_zeta_backward_backward(
                            gg_z, gg_g2, gg_g3, gradient, z, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1),
        iterator.output(2),
        iterator.output(3)
    );
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("weierstrass_zeta", torchscience::cpu::special_functions::weierstrass_zeta);
    module.impl("weierstrass_zeta_backward", torchscience::cpu::special_functions::weierstrass_zeta_backward);
    module.impl("weierstrass_zeta_backward_backward", torchscience::cpu::special_functions::weierstrass_zeta_backward_backward);
}

// ============================================================================
// Custom implementation because weierstrass_eta uses complex operations
// ============================================================================
#include "../kernel/special_functions/weierstrass_eta.h"
#include "../kernel/special_functions/weierstrass_eta_backward.h"
#include "../kernel/special_functions/weierstrass_eta_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor weierstrass_eta(
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor output;
    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_eta",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_eta(g2, g3);
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_eta",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t g2, scalar_t g3) -> scalar_t {
                        return kernel::special_functions::weierstrass_eta(g2, g3);
                    }
                );
            }
        );
    }

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> weierstrass_eta_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor gradient_g2, gradient_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_g2)
        .add_output(gradient_g3)
        .add_input(gradient_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_eta_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_eta_backward(
                            gradient, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_eta_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (scalar_t gradient, scalar_t g2, scalar_t g3)
                        -> std::tuple<scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_eta_backward(
                            gradient, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1)
    );
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> weierstrass_eta_backward_backward(
    const at::Tensor &gg_g2_input,
    const at::Tensor &gg_g3_input,
    const at::Tensor &gradient_input,
    const at::Tensor &g2_input,
    const at::Tensor &g3_input
) {
    at::Tensor grad_grad, grad_g2, grad_g3;
    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad)
        .add_output(grad_g2)
        .add_output(grad_g3)
        .add_input(gg_g2_input)
        .add_input(gg_g3_input)
        .add_input(gradient_input)
        .add_input(g2_input)
        .add_input(g3_input)
        .build();

    if (at::isComplexType(iterator.common_dtype())) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "weierstrass_eta_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_eta_backward_backward(
                            gg_g2, gg_g3, gradient, g2, g3
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "weierstrass_eta_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t gg_g2,
                        scalar_t gg_g3,
                        scalar_t gradient,
                        scalar_t g2,
                        scalar_t g3
                    ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                        return kernel::special_functions::weierstrass_eta_backward_backward(
                            gg_g2, gg_g3, gradient, g2, g3
                        );
                    }
                );
            }
        );
    }

    return std::make_tuple(
        iterator.output(0),
        iterator.output(1),
        iterator.output(2)
    );
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("weierstrass_eta", torchscience::cpu::special_functions::weierstrass_eta);
    module.impl("weierstrass_eta_backward", torchscience::cpu::special_functions::weierstrass_eta_backward);
    module.impl("weierstrass_eta_backward_backward", torchscience::cpu::special_functions::weierstrass_eta_backward_backward);
}

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

#include "../kernel/special_functions/spherical_hankel_1.h"
#include "../kernel/special_functions/spherical_hankel_1_backward.h"
#include "../kernel/special_functions/spherical_hankel_1_backward_backward.h"

// Custom implementation for spherical_hankel_1 since it requires complex output
// The Python wrapper ensures inputs are complex, so we only dispatch to complex types
namespace torchscience::cpu::special_functions {

inline at::Tensor spherical_hankel_1(
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_1",
        [&] {
            at::native::cpu_kernel(
                iterator,
                [] (scalar_t n, scalar_t z) -> scalar_t {
                    return kernel::special_functions::spherical_hankel_1(n, z);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> spherical_hankel_1_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_1_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [] (scalar_t gradient, scalar_t n, scalar_t z) -> std::tuple<scalar_t, scalar_t> {
                    return kernel::special_functions::spherical_hankel_1_backward(gradient, n, z);
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> spherical_hankel_1_backward_backward(
    const at::Tensor &n_gradient_gradient_input,
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    if (!n_gradient_gradient_input.defined() && !z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::Tensor gradient_gradient_output;
    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto n_gg = n_gradient_gradient_input.defined()
        ? n_gradient_gradient_input
        : at::zeros_like(n_input);
    auto z_gg = z_gradient_gradient_input.defined()
        ? z_gradient_gradient_input
        : at::zeros_like(z_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(n_gg)
        .add_const_input(z_gg)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_1_backward_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [] (
                    scalar_t n_gradient_gradient,
                    scalar_t z_gradient_gradient,
                    scalar_t gradient,
                    scalar_t n,
                    scalar_t z
                ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    return kernel::special_functions::spherical_hankel_1_backward_backward(
                        n_gradient_gradient, z_gradient_gradient, gradient, n, z
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("spherical_hankel_1", torchscience::cpu::special_functions::spherical_hankel_1);
    module.impl("spherical_hankel_1_backward", torchscience::cpu::special_functions::spherical_hankel_1_backward);
    module.impl("spherical_hankel_1_backward_backward", torchscience::cpu::special_functions::spherical_hankel_1_backward_backward);
}

#include "../kernel/special_functions/spherical_hankel_2.h"
#include "../kernel/special_functions/spherical_hankel_2_backward.h"
#include "../kernel/special_functions/spherical_hankel_2_backward_backward.h"

// Custom implementation for spherical_hankel_2 since it requires complex output
// The Python wrapper ensures inputs are complex, so we only dispatch to complex types
namespace torchscience::cpu::special_functions {

inline at::Tensor spherical_hankel_2(
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_2",
        [&] {
            at::native::cpu_kernel(
                iterator,
                [] (scalar_t n, scalar_t z) -> scalar_t {
                    return kernel::special_functions::spherical_hankel_2(n, z);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> spherical_hankel_2_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_2_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [] (scalar_t gradient, scalar_t n, scalar_t z) -> std::tuple<scalar_t, scalar_t> {
                    return kernel::special_functions::spherical_hankel_2_backward(gradient, n, z);
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> spherical_hankel_2_backward_backward(
    const at::Tensor &n_gradient_gradient_input,
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &n_input,
    const at::Tensor &z_input
) {
    if (!n_gradient_gradient_input.defined() && !z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::Tensor gradient_gradient_output;
    at::Tensor n_gradient_output;
    at::Tensor z_gradient_output;

    auto n_gg = n_gradient_gradient_input.defined()
        ? n_gradient_gradient_input
        : at::zeros_like(n_input);
    auto z_gg = z_gradient_gradient_input.defined()
        ? z_gradient_gradient_input
        : at::zeros_like(z_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(n_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(n_gg)
        .add_const_input(z_gg)
        .add_const_input(gradient_input)
        .add_const_input(n_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "spherical_hankel_2_backward_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [] (
                    scalar_t n_gradient_gradient,
                    scalar_t z_gradient_gradient,
                    scalar_t gradient,
                    scalar_t n,
                    scalar_t z
                ) -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    return kernel::special_functions::spherical_hankel_2_backward_backward(
                        n_gradient_gradient, z_gradient_gradient, gradient, n, z
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("spherical_hankel_2", torchscience::cpu::special_functions::spherical_hankel_2);
    module.impl("spherical_hankel_2_backward", torchscience::cpu::special_functions::spherical_hankel_2_backward);
    module.impl("spherical_hankel_2_backward_backward", torchscience::cpu::special_functions::spherical_hankel_2_backward_backward);
}

// Airy function of the first kind
#include "../kernel/special_functions/airy_ai.h"
#include "../kernel/special_functions/airy_ai_backward.h"
#include "../kernel/special_functions/airy_ai_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(airy_ai, x)

// Airy function of the second kind
#include "../kernel/special_functions/airy_bi.h"
#include "../kernel/special_functions/airy_bi_backward.h"
#include "../kernel/special_functions/airy_bi_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(airy_bi, x)

// Lambert W function (product logarithm)
#include "../kernel/special_functions/lambert_w.h"
#include "../kernel/special_functions/lambert_w_backward.h"
#include "../kernel/special_functions/lambert_w_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(lambert_w, k, z)

// Kelvin function ber (real part of J_0 at rotated argument)
#include "../kernel/special_functions/kelvin_ber.h"
#include "../kernel/special_functions/kelvin_ber_backward.h"
#include "../kernel/special_functions/kelvin_ber_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(kelvin_ber, x)

// Kelvin function bei (imaginary part of J_0 at rotated argument)
#include "../kernel/special_functions/kelvin_bei.h"
#include "../kernel/special_functions/kelvin_bei_backward.h"
#include "../kernel/special_functions/kelvin_bei_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(kelvin_bei, x)

// Kelvin function ker (real part of K_0 at rotated argument)
#include "../kernel/special_functions/kelvin_ker.h"
#include "../kernel/special_functions/kelvin_ker_backward.h"
#include "../kernel/special_functions/kelvin_ker_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(kelvin_ker, x)

// Kelvin function kei (imaginary part of K_0 at rotated argument)
#include "../kernel/special_functions/kelvin_kei.h"
#include "../kernel/special_functions/kelvin_kei_backward.h"
#include "../kernel/special_functions/kelvin_kei_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(kelvin_kei, x)

// Riemann zeta function (s > 1 only)
#include "../kernel/special_functions/zeta.h"
#include "../kernel/special_functions/zeta_backward.h"
#include "../kernel/special_functions/zeta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(zeta, s)

// Polylogarithm function Li_s(z)
#include "../kernel/special_functions/polylogarithm_li.h"
#include "../kernel/special_functions/polylogarithm_li_backward.h"
#include "../kernel/special_functions/polylogarithm_li_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(polylogarithm_li, s, z)

#include "../kernel/special_functions/parabolic_cylinder_u.h"
#include "../kernel/special_functions/parabolic_cylinder_u_backward.h"
#include "../kernel/special_functions/parabolic_cylinder_u_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(parabolic_cylinder_u, a, z)

#include "../kernel/special_functions/parabolic_cylinder_v.h"
#include "../kernel/special_functions/parabolic_cylinder_v_backward.h"
#include "../kernel/special_functions/parabolic_cylinder_v_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(parabolic_cylinder_v, a, z)

// Faddeeva function w(z) = exp(-z^2) * erfc(-iz)
// Custom implementation since real input produces complex output
#include "../kernel/special_functions/faddeeva_w.h"
#include "../kernel/special_functions/faddeeva_w_backward.h"
#include "../kernel/special_functions/faddeeva_w_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor faddeeva_w(const at::Tensor &z_input) {
    at::Tensor output;

    // For Faddeeva function, output is always complex
    // Promote real inputs to complex
    at::Tensor z_complex;
    if (!z_input.is_complex()) {
        if (z_input.scalar_type() == at::kFloat ||
            z_input.scalar_type() == at::kHalf ||
            z_input.scalar_type() == at::kBFloat16) {
            z_complex = z_input.to(at::kFloat).to(at::kComplexFloat);
        } else {
            z_complex = z_input.to(at::kDouble).to(at::kComplexDouble);
        }
    } else {
        z_complex = z_input;
    }

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(z_complex)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "faddeeva_w",
        [&] {
            at::native::cpu_kernel(
                iterator,
                [] (scalar_t z) -> scalar_t {
                    return kernel::special_functions::faddeeva_w(z);
                }
            );
        }
    );

    return iterator.output();
}

inline at::Tensor faddeeva_w_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    at::Tensor gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "faddeeva_w_backward",
        [&] {
            at::native::cpu_kernel(
                iterator,
                [] (scalar_t gradient, scalar_t z) -> scalar_t {
                    return kernel::special_functions::faddeeva_w_backward(gradient, z);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> faddeeva_w_backward_backward(
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    if (!z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor()};
    }

    at::Tensor gradient_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(z_gradient_gradient_input)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_COMPLEX_TYPES(
        iterator.common_dtype(),
        "faddeeva_w_backward_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [] (
                    scalar_t z_gradient_gradient,
                    scalar_t gradient,
                    scalar_t z
                ) -> std::tuple<scalar_t, scalar_t> {
                    return kernel::special_functions::faddeeva_w_backward_backward(
                        z_gradient_gradient, gradient, z
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1)};
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("faddeeva_w", torchscience::cpu::special_functions::faddeeva_w);
    module.impl("faddeeva_w_backward", torchscience::cpu::special_functions::faddeeva_w_backward);
    module.impl("faddeeva_w_backward_backward", torchscience::cpu::special_functions::faddeeva_w_backward_backward);
}

// Inverse error function
#include "../kernel/special_functions/erfinv.h"
#include "../kernel/special_functions/erfinv_backward.h"
#include "../kernel/special_functions/erfinv_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR(erfinv, x)

// Inverse complementary error function
#include "../kernel/special_functions/erfcinv.h"
#include "../kernel/special_functions/erfcinv_backward.h"
#include "../kernel/special_functions/erfcinv_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR(erfcinv, x)

// Fresnel sine integral
#include "../kernel/special_functions/fresnel_s.h"
#include "../kernel/special_functions/fresnel_s_backward.h"
#include "../kernel/special_functions/fresnel_s_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(fresnel_s, z)

// Fresnel cosine integral
#include "../kernel/special_functions/fresnel_c.h"
#include "../kernel/special_functions/fresnel_c_backward.h"
#include "../kernel/special_functions/fresnel_c_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(fresnel_c, z)

// Dawson's integral
// Custom implementation because dawson uses faddeeva_w internally
// which requires complex operations not supported by Half/BFloat16
#include "../kernel/special_functions/dawson.h"
#include "../kernel/special_functions/dawson_backward.h"
#include "../kernel/special_functions/dawson_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor dawson(const at::Tensor &z_input) {
    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    if (iterator.common_dtype() == at::kComplexFloat ||
        iterator.common_dtype() == at::kComplexDouble) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "dawson",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z) -> scalar_t {
                        return kernel::special_functions::dawson(z);
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "dawson",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t z) -> scalar_t {
                        return kernel::special_functions::dawson(z);
                    }
                );
            }
        );
    }

    return iterator.output();
}

inline at::Tensor dawson_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    at::Tensor gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    if (iterator.common_dtype() == at::kComplexFloat ||
        iterator.common_dtype() == at::kComplexDouble) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "dawson_backward",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t gradient, scalar_t z) -> scalar_t {
                        return kernel::special_functions::dawson_backward(gradient, z);
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "dawson_backward",
            [&] {
                at::native::cpu_kernel(
                    iterator,
                    [] (scalar_t gradient, scalar_t z) -> scalar_t {
                        return kernel::special_functions::dawson_backward(gradient, z);
                    }
                );
            }
        );
    }

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> dawson_backward_backward(
    const at::Tensor &z_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &z_input
) {
    if (!z_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor()};
    }

    at::Tensor gradient_gradient_output;
    at::Tensor z_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(z_gradient_output)
        .add_const_input(z_gradient_gradient_input)
        .add_const_input(gradient_input)
        .add_const_input(z_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    if (iterator.common_dtype() == at::kComplexFloat ||
        iterator.common_dtype() == at::kComplexDouble) {
        AT_DISPATCH_COMPLEX_TYPES(
            iterator.common_dtype(),
            "dawson_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t z_gradient_gradient,
                        scalar_t gradient,
                        scalar_t z
                    ) -> std::tuple<scalar_t, scalar_t> {
                        return kernel::special_functions::dawson_backward_backward(
                            z_gradient_gradient, gradient, z
                        );
                    }
                );
            }
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES(
            iterator.common_dtype(),
            "dawson_backward_backward",
            [&] {
                at::native::cpu_kernel_multiple_outputs(
                    iterator,
                    [] (
                        scalar_t z_gradient_gradient,
                        scalar_t gradient,
                        scalar_t z
                    ) -> std::tuple<scalar_t, scalar_t> {
                        return kernel::special_functions::dawson_backward_backward(
                            z_gradient_gradient, gradient, z
                        );
                    }
                );
            }
        );
    }

    return {iterator.output(0), iterator.output(1)};
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("dawson", torchscience::cpu::special_functions::dawson);
    module.impl("dawson_backward", torchscience::cpu::special_functions::dawson_backward);
    module.impl("dawson_backward_backward", torchscience::cpu::special_functions::dawson_backward_backward);
}

// Voigt profile
// Custom implementation because voigt_profile uses faddeeva_w internally
// which requires complex operations not supported by Half/BFloat16
#include "../kernel/special_functions/voigt_profile.h"
#include "../kernel/special_functions/voigt_profile_backward.h"
#include "../kernel/special_functions/voigt_profile_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor voigt_profile(
    const at::Tensor &x_input,
    const at::Tensor &sigma_input,
    const at::Tensor &gamma_input
) {
    at::Tensor output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_const_input(x_input)
        .add_const_input(sigma_input)
        .add_const_input(gamma_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES(
        iterator.common_dtype(),
        "voigt_profile",
        [&] {
            at::native::cpu_kernel(
                iterator,
                [] (scalar_t x, scalar_t sigma, scalar_t gamma) -> scalar_t {
                    return kernel::special_functions::voigt_profile(x, sigma, gamma);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> voigt_profile_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &x_input,
    const at::Tensor &sigma_input,
    const at::Tensor &gamma_input
) {
    at::Tensor x_gradient_output;
    at::Tensor sigma_gradient_output;
    at::Tensor gamma_gradient_output;

    auto iterator = at::TensorIteratorConfig()
        .add_output(x_gradient_output)
        .add_output(sigma_gradient_output)
        .add_output(gamma_gradient_output)
        .add_const_input(gradient_input)
        .add_const_input(x_input)
        .add_const_input(sigma_input)
        .add_const_input(gamma_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES(
        iterator.common_dtype(),
        "voigt_profile_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [] (scalar_t gradient, scalar_t x, scalar_t sigma, scalar_t gamma)
                    -> std::tuple<scalar_t, scalar_t, scalar_t> {
                    return kernel::special_functions::voigt_profile_backward(
                        gradient, x, sigma, gamma
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2)};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> voigt_profile_backward_backward(
    const at::Tensor &x_gradient_gradient_input,
    const at::Tensor &sigma_gradient_gradient_input,
    const at::Tensor &gamma_gradient_gradient_input,
    const at::Tensor &gradient_input,
    const at::Tensor &x_input,
    const at::Tensor &sigma_input,
    const at::Tensor &gamma_input
) {
    if (!x_gradient_gradient_input.defined() &&
        !sigma_gradient_gradient_input.defined() &&
        !gamma_gradient_gradient_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    at::Tensor gradient_gradient_output;
    at::Tensor x_gradient_output;
    at::Tensor sigma_gradient_output;
    at::Tensor gamma_gradient_output;

    auto x_gg = x_gradient_gradient_input.defined()
        ? x_gradient_gradient_input
        : at::zeros_like(x_input);
    auto sigma_gg = sigma_gradient_gradient_input.defined()
        ? sigma_gradient_gradient_input
        : at::zeros_like(sigma_input);
    auto gamma_gg = gamma_gradient_gradient_input.defined()
        ? gamma_gradient_gradient_input
        : at::zeros_like(gamma_input);

    auto iterator = at::TensorIteratorConfig()
        .add_output(gradient_gradient_output)
        .add_output(x_gradient_output)
        .add_output(sigma_gradient_output)
        .add_output(gamma_gradient_output)
        .add_const_input(x_gg)
        .add_const_input(sigma_gg)
        .add_const_input(gamma_gg)
        .add_const_input(gradient_input)
        .add_const_input(x_input)
        .add_const_input(sigma_input)
        .add_const_input(gamma_input)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .build();

    AT_DISPATCH_FLOATING_TYPES(
        iterator.common_dtype(),
        "voigt_profile_backward_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [] (
                    scalar_t gg_x,
                    scalar_t gg_sigma,
                    scalar_t gg_gamma,
                    scalar_t gradient,
                    scalar_t x,
                    scalar_t sigma,
                    scalar_t gamma
                ) -> std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> {
                    return kernel::special_functions::voigt_profile_backward_backward(
                        gg_x, gg_sigma, gg_gamma, gradient, x, sigma, gamma
                    );
                }
            );
        }
    );

    return {iterator.output(0), iterator.output(1), iterator.output(2), iterator.output(3)};
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("voigt_profile", torchscience::cpu::special_functions::voigt_profile);
    module.impl("voigt_profile_backward", torchscience::cpu::special_functions::voigt_profile_backward);
    module.impl("voigt_profile_backward_backward", torchscience::cpu::special_functions::voigt_profile_backward_backward);
}

// Generalized hypergeometric function pFq
// a has shape [..., p], b has shape [..., q], z has shape [...], output has shape [...]
#include "../kernel/special_functions/hypergeometric_p_f_q.h"
#include "../kernel/special_functions/hypergeometric_p_f_q_backward.h"
#include "../kernel/special_functions/hypergeometric_p_f_q_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor hypergeometric_p_f_q(
    const at::Tensor &a_input,
    const at::Tensor &b_input,
    const at::Tensor &z_input
) {
    // Get dimensions
    TORCH_CHECK(a_input.dim() >= 1, "a must have at least 1 dimension");
    TORCH_CHECK(b_input.dim() >= 1, "b must have at least 1 dimension");

    int64_t p = a_input.size(-1);
    int64_t q = b_input.size(-1);

    // Get batch shapes (all dimensions except last)
    auto a_batch = a_input.sizes().slice(0, a_input.dim() - 1);
    auto b_batch = b_input.sizes().slice(0, b_input.dim() - 1);
    auto z_batch = z_input.sizes();

    // Ensure inputs have same dtype
    auto ab_dtype = at::result_type(a_input, b_input);
    auto common_dtype = at::promote_types(ab_dtype, z_input.scalar_type());
    auto a = a_input.to(common_dtype).contiguous();
    auto b = b_input.to(common_dtype).contiguous();
    auto z = z_input.to(common_dtype).contiguous();

    // Broadcast batch dimensions
    std::vector<int64_t> output_shape;
    int64_t max_batch_dim = std::max({a_input.dim() - 1, b_input.dim() - 1, z_input.dim()});

    // Compute broadcasted shape
    for (int64_t i = 0; i < max_batch_dim; ++i) {
        int64_t a_size = (i < static_cast<int64_t>(a_batch.size())) ? a_batch[a_batch.size() - 1 - i] : 1;
        int64_t b_size = (i < static_cast<int64_t>(b_batch.size())) ? b_batch[b_batch.size() - 1 - i] : 1;
        int64_t z_size = (i < z_input.dim()) ? z_batch[z_input.dim() - 1 - i] : 1;

        int64_t max_size = std::max({a_size, b_size, z_size});
        output_shape.insert(output_shape.begin(), max_size);
    }

    // Track if output should be scalar
    bool output_is_scalar = output_shape.empty();
    std::vector<int64_t> working_shape = output_shape;
    if (working_shape.empty()) {
        working_shape.push_back(1);  // For computation, treat as [1]
    }

    // Create output tensor with working shape
    auto output = at::empty(working_shape, a.options());

    // Flatten for iteration
    int64_t batch_size = 1;
    for (auto s : working_shape) {
        batch_size *= s;
    }

    // Reshape tensors for iteration
    // First, add leading singleton dimensions to match working_shape rank
    int64_t working_rank = static_cast<int64_t>(working_shape.size());

    // For a: needs working_rank + 1 dimensions (batch dims + p)
    std::vector<int64_t> a_view_shape(working_rank + 1, 1);
    for (int64_t i = 0; i < a.dim() - 1; ++i) {
        a_view_shape[working_rank - (a.dim() - 1) + i] = a.size(i);
    }
    a_view_shape[working_rank] = p;
    auto a_reshaped = a.view(a_view_shape);

    // For b: needs working_rank + 1 dimensions (batch dims + q)
    std::vector<int64_t> b_view_shape(working_rank + 1, 1);
    for (int64_t i = 0; i < b.dim() - 1; ++i) {
        b_view_shape[working_rank - (b.dim() - 1) + i] = b.size(i);
    }
    b_view_shape[working_rank] = q;
    auto b_reshaped = b.view(b_view_shape);

    // For z: needs working_rank dimensions (batch dims only)
    std::vector<int64_t> z_view_shape(working_rank, 1);
    for (int64_t i = 0; i < z.dim(); ++i) {
        z_view_shape[working_rank - z.dim() + i] = z.size(i);
    }
    auto z_reshaped = z.view(z_view_shape);

    // Build target shapes for expansion
    std::vector<int64_t> a_expanded_shape = working_shape;
    a_expanded_shape.push_back(p);
    std::vector<int64_t> b_expanded_shape = working_shape;
    b_expanded_shape.push_back(q);

    auto a_expanded = a_reshaped.expand(a_expanded_shape).contiguous().view({batch_size, p});
    auto b_expanded = b_reshaped.expand(b_expanded_shape).contiguous().view({batch_size, q});
    auto z_expanded = z_reshaped.expand(working_shape).contiguous().view({batch_size});
    auto output_flat = output.view({batch_size});

    // Dispatch based on dtype
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kComplexFloat, at::kComplexDouble,
        common_dtype,
        "hypergeometric_p_f_q",
        [&] {
            auto a_ptr = a_expanded.data_ptr<scalar_t>();
            auto b_ptr = b_expanded.data_ptr<scalar_t>();
            auto z_ptr = z_expanded.data_ptr<scalar_t>();
            auto out_ptr = output_flat.data_ptr<scalar_t>();

            for (int64_t i = 0; i < batch_size; ++i) {
                out_ptr[i] = kernel::special_functions::hypergeometric_p_f_q(
                    a_ptr + i * p, static_cast<int>(p),
                    b_ptr + i * q, static_cast<int>(q),
                    z_ptr[i]
                );
            }
        }
    );

    // Return scalar if output should be scalar
    if (output_is_scalar) {
        return output.squeeze();
    }
    return output;
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> hypergeometric_p_f_q_backward(
    const at::Tensor &grad_input,
    const at::Tensor &a_input,
    const at::Tensor &b_input,
    const at::Tensor &z_input
) {
    TORCH_CHECK(a_input.dim() >= 1, "a must have at least 1 dimension");
    TORCH_CHECK(b_input.dim() >= 1, "b must have at least 1 dimension");

    int64_t p = a_input.size(-1);
    int64_t q = b_input.size(-1);

    // Ensure inputs have same dtype
    auto ab_dtype = at::result_type(a_input, b_input);
    auto common_dtype = at::promote_types(ab_dtype, z_input.scalar_type());
    auto grad = grad_input.to(common_dtype).contiguous();
    auto a = a_input.to(common_dtype).contiguous();
    auto b = b_input.to(common_dtype).contiguous();
    auto z = z_input.to(common_dtype).contiguous();

    // Get output shapes (same as inputs)
    auto grad_a = at::empty_like(a);
    auto grad_b = at::empty_like(b);
    auto grad_z = at::empty_like(z);

    // Flatten for iteration
    int64_t batch_size = grad.numel();

    auto grad_flat = grad.view({batch_size});
    auto a_flat = a.view({-1, p});
    auto b_flat = b.view({-1, q});
    auto z_flat = z.view({batch_size});
    auto grad_a_flat = grad_a.view({-1, p});
    auto grad_b_flat = grad_b.view({-1, q});
    auto grad_z_flat = grad_z.view({batch_size});

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kComplexFloat, at::kComplexDouble,
        common_dtype,
        "hypergeometric_p_f_q_backward",
        [&] {
            auto grad_ptr = grad_flat.data_ptr<scalar_t>();
            auto a_ptr = a_flat.data_ptr<scalar_t>();
            auto b_ptr = b_flat.data_ptr<scalar_t>();
            auto z_ptr = z_flat.data_ptr<scalar_t>();
            auto grad_a_ptr = grad_a_flat.data_ptr<scalar_t>();
            auto grad_b_ptr = grad_b_flat.data_ptr<scalar_t>();
            auto grad_z_ptr = grad_z_flat.data_ptr<scalar_t>();

            for (int64_t i = 0; i < batch_size; ++i) {
                kernel::special_functions::hypergeometric_p_f_q_backward(
                    grad_ptr[i],
                    a_ptr + i * p, static_cast<int>(p),
                    b_ptr + i * q, static_cast<int>(q),
                    z_ptr[i],
                    grad_a_ptr + i * p,
                    grad_b_ptr + i * q,
                    grad_z_ptr[i]
                );
            }
        }
    );

    return {grad_a, grad_b, grad_z};
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hypergeometric_p_f_q_backward_backward(
    const at::Tensor &gg_a_input,
    const at::Tensor &gg_b_input,
    const at::Tensor &gg_z_input,
    const at::Tensor &grad_input,
    const at::Tensor &a_input,
    const at::Tensor &b_input,
    const at::Tensor &z_input
) {
    if (!gg_a_input.defined() && !gg_b_input.defined() && !gg_z_input.defined()) {
        return {at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }

    int64_t p = a_input.size(-1);
    int64_t q = b_input.size(-1);

    auto ab_dtype = at::result_type(a_input, b_input);
    auto common_dtype = at::promote_types(ab_dtype, z_input.scalar_type());
    auto gg_a = gg_a_input.defined() ? gg_a_input.to(common_dtype).contiguous() : at::zeros_like(a_input);
    auto gg_b = gg_b_input.defined() ? gg_b_input.to(common_dtype).contiguous() : at::zeros_like(b_input);
    auto gg_z = gg_z_input.defined() ? gg_z_input.to(common_dtype).contiguous() : at::zeros_like(z_input);
    auto grad = grad_input.to(common_dtype).contiguous();
    auto a = a_input.to(common_dtype).contiguous();
    auto b = b_input.to(common_dtype).contiguous();
    auto z = z_input.to(common_dtype).contiguous();

    auto grad_grad = at::empty_like(grad);
    auto out_grad_a = at::empty_like(a);
    auto out_grad_b = at::empty_like(b);
    auto out_grad_z = at::empty_like(z);

    int64_t batch_size = grad.numel();

    auto gg_a_flat = gg_a.view({-1, p});
    auto gg_b_flat = gg_b.view({-1, q});
    auto gg_z_flat = gg_z.view({batch_size});
    auto grad_flat = grad.view({batch_size});
    auto a_flat = a.view({-1, p});
    auto b_flat = b.view({-1, q});
    auto z_flat = z.view({batch_size});
    auto grad_grad_flat = grad_grad.view({batch_size});
    auto out_grad_a_flat = out_grad_a.view({-1, p});
    auto out_grad_b_flat = out_grad_b.view({-1, q});
    auto out_grad_z_flat = out_grad_z.view({batch_size});

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kComplexFloat, at::kComplexDouble,
        common_dtype,
        "hypergeometric_p_f_q_backward_backward",
        [&] {
            auto gg_a_ptr = gg_a_flat.data_ptr<scalar_t>();
            auto gg_b_ptr = gg_b_flat.data_ptr<scalar_t>();
            auto gg_z_ptr = gg_z_flat.data_ptr<scalar_t>();
            auto grad_ptr = grad_flat.data_ptr<scalar_t>();
            auto a_ptr = a_flat.data_ptr<scalar_t>();
            auto b_ptr = b_flat.data_ptr<scalar_t>();
            auto z_ptr = z_flat.data_ptr<scalar_t>();
            auto grad_grad_ptr = grad_grad_flat.data_ptr<scalar_t>();
            auto out_grad_a_ptr = out_grad_a_flat.data_ptr<scalar_t>();
            auto out_grad_b_ptr = out_grad_b_flat.data_ptr<scalar_t>();
            auto out_grad_z_ptr = out_grad_z_flat.data_ptr<scalar_t>();

            for (int64_t i = 0; i < batch_size; ++i) {
                kernel::special_functions::hypergeometric_p_f_q_backward_backward(
                    gg_a_ptr + i * p, static_cast<int>(p),
                    gg_b_ptr + i * q, static_cast<int>(q),
                    gg_z_ptr[i],
                    grad_ptr[i],
                    a_ptr + i * p,
                    b_ptr + i * q,
                    z_ptr[i],
                    grad_grad_ptr[i],
                    out_grad_a_ptr + i * p,
                    out_grad_b_ptr + i * q,
                    out_grad_z_ptr[i]
                );
            }
        }
    );

    return {grad_grad, out_grad_a, out_grad_b, out_grad_z};
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("hypergeometric_p_f_q", torchscience::cpu::special_functions::hypergeometric_p_f_q);
    module.impl("hypergeometric_p_f_q_backward", torchscience::cpu::special_functions::hypergeometric_p_f_q_backward);
    module.impl("hypergeometric_p_f_q_backward_backward", torchscience::cpu::special_functions::hypergeometric_p_f_q_backward_backward);
}

// Legendre polynomial P_n(z)
#include "../kernel/special_functions/legendre_polynomial_p.h"
#include "../kernel/special_functions/legendre_polynomial_p_backward.h"
#include "../kernel/special_functions/legendre_polynomial_p_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(legendre_polynomial_p, n, z)

// Legendre function of the second kind Q_n(x)
#include "../kernel/special_functions/legendre_polynomial_q.h"
#include "../kernel/special_functions/legendre_polynomial_q_backward.h"
#include "../kernel/special_functions/legendre_polynomial_q_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(legendre_polynomial_q, x, n)

// Hermite polynomial (physicists') H_n(z)
#include "../kernel/special_functions/hermite_polynomial_h.h"
#include "../kernel/special_functions/hermite_polynomial_h_backward.h"
#include "../kernel/special_functions/hermite_polynomial_h_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(hermite_polynomial_h, n, z)

// Hermite polynomial (probabilists') He_n(z)
#include "../kernel/special_functions/hermite_polynomial_he.h"
#include "../kernel/special_functions/hermite_polynomial_he_backward.h"
#include "../kernel/special_functions/hermite_polynomial_he_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(hermite_polynomial_he, n, z)

// Generalized Laguerre polynomial L_n^alpha(z)
#include "../kernel/special_functions/laguerre_polynomial_l.h"
#include "../kernel/special_functions/laguerre_polynomial_l_backward.h"
#include "../kernel/special_functions/laguerre_polynomial_l_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(laguerre_polynomial_l, n, alpha, z)

// Gegenbauer (ultraspherical) polynomial C_n^lambda(z)
#include "../kernel/special_functions/gegenbauer_polynomial_c.h"
#include "../kernel/special_functions/gegenbauer_polynomial_c_backward.h"
#include "../kernel/special_functions/gegenbauer_polynomial_c_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(gegenbauer_polynomial_c, n, lambda, z)

// Jacobi polynomial P_n^(alpha,beta)(z)
#include "../kernel/special_functions/jacobi_polynomial_p.h"
#include "../kernel/special_functions/jacobi_polynomial_p_backward.h"
#include "../kernel/special_functions/jacobi_polynomial_p_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(jacobi_polynomial_p, n, alpha, beta, z)

// Radial Zernike polynomial R_n^m(rho)
#include "../kernel/special_functions/zernike_polynomial_r.h"
#include "../kernel/special_functions/zernike_polynomial_r_backward.h"
#include "../kernel/special_functions/zernike_polynomial_r_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(zernike_polynomial_r, n, m, rho)

// Full Zernike polynomial Z_n^m(rho, theta)
#include "../kernel/special_functions/zernike_polynomial_z.h"
#include "../kernel/special_functions/zernike_polynomial_z_backward.h"
#include "../kernel/special_functions/zernike_polynomial_z_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(zernike_polynomial_z, n, m, rho, theta)

// Krawtchouk polynomial K_n(x; p, N)
#include "../kernel/special_functions/krawtchouk_polynomial_k.h"
#include "../kernel/special_functions/krawtchouk_polynomial_k_backward.h"
#include "../kernel/special_functions/krawtchouk_polynomial_k_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(krawtchouk_polynomial_k, n, x, p, N)

// Meixner polynomial M_n(x; beta, c)
#include "../kernel/special_functions/meixner_polynomial_m.h"
#include "../kernel/special_functions/meixner_polynomial_m_backward.h"
#include "../kernel/special_functions/meixner_polynomial_m_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUATERNARY_OPERATOR_WITH_COMPLEX(meixner_polynomial_m, n, x, beta, c)

// Hahn polynomial Q_n(x; alpha, beta, N)
#include "../kernel/special_functions/hahn_polynomial_q.h"
#include "../kernel/special_functions/hahn_polynomial_q_backward.h"
#include "../kernel/special_functions/hahn_polynomial_q_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_QUINARY_OPERATOR(hahn_polynomial_q, n, x, alpha, beta, N)

// Charlier polynomial C_n(x; a)
#include "../kernel/special_functions/charlier_polynomial_c.h"
#include "../kernel/special_functions/charlier_polynomial_c_backward.h"
#include "../kernel/special_functions/charlier_polynomial_c_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR_WITH_COMPLEX(charlier_polynomial_c, n, x, a)

// Pochhammer symbol (rising factorial)
#include "../kernel/special_functions/pochhammer.h"
#include "../kernel/special_functions/pochhammer_backward.h"
#include "../kernel/special_functions/pochhammer_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(pochhammer, z, m)

// Log multivariate gamma function
// Custom implementation because d is an integer parameter, not a tensor
#include "../kernel/special_functions/log_multivariate_gamma.h"
#include "../kernel/special_functions/log_multivariate_gamma_backward.h"
#include "../kernel/special_functions/log_multivariate_gamma_backward_backward.h"

namespace torchscience::cpu::special_functions {

inline at::Tensor log_multivariate_gamma(const at::Tensor &a_input, int64_t d) {
    at::Tensor output;
    auto iterator = at::TensorIteratorConfig()
        .add_output(output)
        .add_input(a_input)
        .build();

    AT_DISPATCH_FLOATING_TYPES(
        iterator.common_dtype(),
        "log_multivariate_gamma",
        [&] {
            at::native::cpu_kernel(
                iterator,
                [d] (scalar_t a) -> scalar_t {
                    return kernel::special_functions::log_multivariate_gamma(a, d);
                }
            );
        }
    );

    return iterator.output();
}

inline at::Tensor log_multivariate_gamma_backward(
    const at::Tensor &gradient_input,
    const at::Tensor &a_input,
    int64_t d
) {
    at::Tensor grad_a;
    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_a)
        .add_input(gradient_input)
        .add_input(a_input)
        .build();

    AT_DISPATCH_FLOATING_TYPES(
        iterator.common_dtype(),
        "log_multivariate_gamma_backward",
        [&] {
            at::native::cpu_kernel(
                iterator,
                [d] (scalar_t gradient, scalar_t a) -> scalar_t {
                    return kernel::special_functions::log_multivariate_gamma_backward(gradient, a, d);
                }
            );
        }
    );

    return iterator.output();
}

inline std::tuple<at::Tensor, at::Tensor> log_multivariate_gamma_backward_backward(
    const at::Tensor &gg_a_input,
    const at::Tensor &gradient_input,
    const at::Tensor &a_input,
    int64_t d
) {
    at::Tensor grad_grad_output, grad_a;
    auto iterator = at::TensorIteratorConfig()
        .add_output(grad_grad_output)
        .add_output(grad_a)
        .add_input(gg_a_input)
        .add_input(gradient_input)
        .add_input(a_input)
        .build();

    AT_DISPATCH_FLOATING_TYPES(
        iterator.common_dtype(),
        "log_multivariate_gamma_backward_backward",
        [&] {
            at::native::cpu_kernel_multiple_outputs(
                iterator,
                [d] (scalar_t gg_a, scalar_t gradient, scalar_t a)
                    -> std::tuple<scalar_t, scalar_t> {
                    return kernel::special_functions::log_multivariate_gamma_backward_backward(
                        gg_a, gradient, a, d
                    );
                }
            );
        }
    );

    return std::make_tuple(iterator.output(0), iterator.output(1));
}

} // namespace torchscience::cpu::special_functions

TORCH_LIBRARY_IMPL(torchscience, CPU, module) {
    module.impl("log_multivariate_gamma", torchscience::cpu::special_functions::log_multivariate_gamma);
    module.impl("log_multivariate_gamma_backward", torchscience::cpu::special_functions::log_multivariate_gamma_backward);
    module.impl("log_multivariate_gamma_backward_backward", torchscience::cpu::special_functions::log_multivariate_gamma_backward_backward);
}

// Inverse regularized gamma P function
#include "../kernel/special_functions/inverse_regularized_gamma_p.h"
#include "../kernel/special_functions/inverse_regularized_gamma_p_backward.h"
#include "../kernel/special_functions/inverse_regularized_gamma_p_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_p, a, y)

// Inverse regularized gamma Q function
#include "../kernel/special_functions/inverse_regularized_gamma_q.h"
#include "../kernel/special_functions/inverse_regularized_gamma_q_backward.h"
#include "../kernel/special_functions/inverse_regularized_gamma_q_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_q, a, y)

// Inverse regularized incomplete beta function
#include "../kernel/special_functions/inverse_regularized_incomplete_beta.h"
#include "../kernel/special_functions/inverse_regularized_incomplete_beta_backward.h"
#include "../kernel/special_functions/inverse_regularized_incomplete_beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR(inverse_regularized_incomplete_beta, a, b, y)

// Inverse complementary regularized incomplete beta function
#include "../kernel/special_functions/inverse_complementary_regularized_incomplete_beta.h"
#include "../kernel/special_functions/inverse_complementary_regularized_incomplete_beta_backward.h"
#include "../kernel/special_functions/inverse_complementary_regularized_incomplete_beta_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_TERNARY_OPERATOR(inverse_complementary_regularized_incomplete_beta, a, b, y)

// Struve function H_0
#include "../kernel/special_functions/struve_h_0.h"
#include "../kernel/special_functions/struve_h_0_backward.h"
#include "../kernel/special_functions/struve_h_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(struve_h_0, z)

// Struve function H_1
#include "../kernel/special_functions/struve_h_1.h"
#include "../kernel/special_functions/struve_h_1_backward.h"
#include "../kernel/special_functions/struve_h_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(struve_h_1, z)

// Modified Struve function L_0
#include "../kernel/special_functions/struve_l_0.h"
#include "../kernel/special_functions/struve_l_0_backward.h"
#include "../kernel/special_functions/struve_l_0_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(struve_l_0, z)

// Modified Struve function L_1
#include "../kernel/special_functions/struve_l_1.h"
#include "../kernel/special_functions/struve_l_1_backward.h"
#include "../kernel/special_functions/struve_l_1_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_UNARY_OPERATOR_WITH_COMPLEX(struve_l_1, z)

// General order Struve function H_n(z)
#include "../kernel/special_functions/struve_h.h"
#include "../kernel/special_functions/struve_h_backward.h"
#include "../kernel/special_functions/struve_h_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(struve_h, n, z)

// General order modified Struve function L_n(z)
#include "../kernel/special_functions/struve_l.h"
#include "../kernel/special_functions/struve_l_backward.h"
#include "../kernel/special_functions/struve_l_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR_WITH_COMPLEX(struve_l, n, z)

// Anger function J_nu(z)
#include "../kernel/special_functions/anger_j.h"
#include "../kernel/special_functions/anger_j_backward.h"
#include "../kernel/special_functions/anger_j_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(anger_j, n, z)

// Weber function E_nu(z)
#include "../kernel/special_functions/weber_e.h"
#include "../kernel/special_functions/weber_e_backward.h"
#include "../kernel/special_functions/weber_e_backward_backward.h"

TORCHSCIENCE_CPU_POINTWISE_BINARY_OPERATOR(weber_e, n, z)
