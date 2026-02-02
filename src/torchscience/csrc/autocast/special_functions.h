#pragma once

#include "macros.h"

TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(gamma, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(digamma, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(trigamma, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(beta, a, b)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_t, x, n)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_u, x, n)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(incomplete_beta, x, a, b)
TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_2_f_1, a, b, c, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_m, a, b, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_u, a, b, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(polygamma, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(log_beta, a, b)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(log_gamma, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(reciprocal_gamma, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(gamma_sign, x)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(regularized_gamma_p, a, x)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(regularized_gamma_q, a, x)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(modified_bessel_i_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(modified_bessel_i_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(bessel_j_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(bessel_j_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(bessel_y_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(bessel_y_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(modified_bessel_k_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(modified_bessel_k_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(bessel_j, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(bessel_y, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(modified_bessel_k, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(modified_bessel_i, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_f, x, y, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_d, x, y, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_c, x, y)
TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(carlson_elliptic_integral_r_j, x, y, z, p)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_g, x, y, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_e, x, y, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_m, x, y, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_k, x, y)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_k, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_e, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_e, phi, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_f, phi, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(complete_legendre_elliptic_integral_pi, n, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(incomplete_legendre_elliptic_integral_pi, n, phi, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_amplitude_am, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dn, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cn, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sn, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sd, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cd, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sc, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nd, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nc, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ns, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dc, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ds, u, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cs, u, m)

// Inverse Jacobi elliptic functions (primary)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sn, x, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cn, x, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_dn, x, m)

// Inverse Jacobi elliptic functions (derived)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sd, x, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cd, x, m)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sc, x, m)

// Jacobi theta functions
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(theta_1, z, q)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(theta_2, z, q)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(theta_3, z, q)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(theta_4, z, q)

// Weierstrass elliptic function P
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(weierstrass_p, z, g2, g3)

// Weierstrass sigma function
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(weierstrass_sigma, z, g2, g3)

// Weierstrass zeta function
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(weierstrass_zeta, z, g2, g3)

// Weierstrass eta quasi-period
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(weierstrass_eta, g2, g3)

TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(spherical_bessel_j, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(spherical_bessel_y, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(spherical_bessel_i, n, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_0, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_1, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(spherical_bessel_k, n, z)

// Exponential integrals
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(exponential_integral_ei, x)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(exponential_integral_e_1, x)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(exponential_integral_ein, x)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(exponential_integral_e, n, x)

// Sine integral
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(sine_integral_si, x)

// Cosine integral
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(cosine_integral_ci, x)

// Spherical Hankel functions of the first kind
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(spherical_hankel_1, n, z)

// Spherical Hankel functions of the second kind
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(spherical_hankel_2, n, z)

// Airy function of the first kind
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(airy_ai, x)

// Airy function of the second kind
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(airy_bi, x)

// Lambert W function (product logarithm)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(lambert_w, k, z)

// Kelvin function ber (real part of J_0 at rotated argument)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(kelvin_ber, x)

// Kelvin function bei (imaginary part of J_0 at rotated argument)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(kelvin_bei, x)

// Kelvin function ker (real part of K_0 at rotated argument)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(kelvin_ker, x)

// Kelvin function kei (imaginary part of K_0 at rotated argument)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(kelvin_kei, x)

// Riemann zeta function (s > 1 only)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(zeta, s)

// Polylogarithm function Li_s(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(polylogarithm_li, s, z)

TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_u, a, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_v, a, z)

// Whittaker functions
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(whittaker_m, kappa, mu, z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(whittaker_w, kappa, mu, z)

// Hypergeometric 0F1 (confluent hypergeometric limit function)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(hypergeometric_0_f_1, b, z)

// Hypergeometric 1F2
TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_1_f_2, a, b1, b2, z)

// Faddeeva function w(z) = exp(-z^2) * erfc(-iz)
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(faddeeva_w, z)

// Inverse error function
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(erfinv, x)

// Inverse complementary error function
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(erfcinv, x)

// Fresnel sine integral
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(fresnel_s, z)

// Fresnel cosine integral
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(fresnel_c, z)

// Dawson's integral
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(dawson, z)

// Voigt profile
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(voigt_profile, x, sigma, gamma)

// Generalized hypergeometric pFq - custom autocast implementation
namespace torchscience::autocast::special_functions {

inline at::Tensor hypergeometric_p_f_q(
    const at::Tensor &a,
    const at::Tensor &b,
    const at::Tensor &z
) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastCPU);

    auto exec_type = at::autocast::get_autocast_dtype(at::kCPU);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::hypergeometric_p_f_q", "")
        .typed<at::Tensor(const at::Tensor&, const at::Tensor&, const at::Tensor&)>()
        .call(
            at::autocast::cached_cast(exec_type, a),
            at::autocast::cached_cast(exec_type, b),
            at::autocast::cached_cast(exec_type, z)
        );
}

} // namespace torchscience::autocast::special_functions

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("hypergeometric_p_f_q", torchscience::autocast::special_functions::hypergeometric_p_f_q);
}

// Legendre polynomial P_n(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(legendre_polynomial_p, n, z)

// Legendre function of the second kind Q_n(x)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(legendre_polynomial_q, x, n)

// Hermite polynomial (physicists') H_n(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(hermite_polynomial_h, n, z)

// Hermite polynomial (probabilists') He_n(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(hermite_polynomial_he, n, z)

// Generalized Laguerre polynomial L_n^alpha(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(laguerre_polynomial_l, n, alpha, z)

// Gegenbauer (ultraspherical) polynomial C_n^lambda(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(gegenbauer_polynomial_c, n, lambda, z)

// Jacobi polynomial P_n^(alpha,beta)(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(jacobi_polynomial_p, n, alpha, beta, z)

// Radial Zernike polynomial R_n^m(rho)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(zernike_polynomial_r, n, m, rho)

// Full Zernike polynomial Z_n^m(rho, theta)
TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(zernike_polynomial_z, n, m, rho, theta)

// Krawtchouk polynomial K_n(x; p, N)
TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(krawtchouk_polynomial_k, n, x, p, N)

// Meixner polynomial M_n(x; beta, c)
TORCHSCIENCE_AUTOCAST_POINTWISE_QUATERNARY_OPERATOR(meixner_polynomial_m, n, x, beta, c)

// Hahn polynomial Q_n(x; alpha, beta, N)
TORCHSCIENCE_AUTOCAST_POINTWISE_QUINARY_OPERATOR(hahn_polynomial_q, n, x, alpha, beta, N)

// Charlier polynomial C_n(x; a)
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(charlier_polynomial_c, n, x, a)

// Pochhammer symbol (rising factorial)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(pochhammer, z, m)

// Log multivariate gamma autocast
namespace torchscience::autocast::special_functions {

inline at::Tensor log_multivariate_gamma(const at::Tensor &a, int64_t d) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastCPU);

    return c10::Dispatcher::singleton()
        .findSchemaOrThrow("torchscience::log_multivariate_gamma", "")
        .typed<at::Tensor(const at::Tensor&, int64_t)>()
        .call(at::autocast::cached_cast(at::kFloat, a, c10::DeviceType::CPU), d);
}

} // namespace torchscience::autocast::special_functions

TORCH_LIBRARY_IMPL(torchscience, AutocastCPU, module) {
    module.impl("log_multivariate_gamma", torchscience::autocast::special_functions::log_multivariate_gamma);
}

// Inverse regularized gamma P function
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_p, a, y)

// Inverse regularized gamma Q function
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(inverse_regularized_gamma_q, a, y)

// Inverse regularized incomplete beta function
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(inverse_regularized_incomplete_beta, a, b, y)

// Inverse complementary regularized incomplete beta function
TORCHSCIENCE_AUTOCAST_POINTWISE_TERNARY_OPERATOR(inverse_complementary_regularized_incomplete_beta, a, b, y)

// Struve function H_0
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(struve_h_0, z)

// Struve function H_1
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(struve_h_1, z)

// Modified Struve function L_0
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(struve_l_0, z)

// Modified Struve function L_1
TORCHSCIENCE_AUTOCAST_POINTWISE_UNARY_OPERATOR(struve_l_1, z)

// General order Struve function H_n(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(struve_h, n, z)

// General order modified Struve function L_n(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(struve_l, n, z)

// Anger function J_nu(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(anger_j, n, z)

// Weber function E_nu(z)
TORCHSCIENCE_AUTOCAST_POINTWISE_BINARY_OPERATOR(weber_e, n, z)
