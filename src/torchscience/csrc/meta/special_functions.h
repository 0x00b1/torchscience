#pragma once

#include "macros.h"

TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(gamma, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(digamma, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(trigamma, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(beta, a, b)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(chebyshev_polynomial_t, x, n)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(incomplete_beta, x, a, b)
TORCHSCIENCE_META_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_2_f_1, a, b, c, z)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_m, a, b, z)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(confluent_hypergeometric_u, a, b, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(polygamma, n, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(log_beta, a, b)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(log_gamma, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(regularized_gamma_p, a, x)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(regularized_gamma_q, a, x)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(modified_bessel_i_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(modified_bessel_i_1, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(bessel_j_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(bessel_j_1, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(bessel_y_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(bessel_y_1, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(modified_bessel_k_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(modified_bessel_k_1, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(bessel_j, n, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(bessel_y, n, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(modified_bessel_k, n, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(modified_bessel_i, n, z)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_f, x, y, z)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_d, x, y, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_c, x, y)
TORCHSCIENCE_META_POINTWISE_QUATERNARY_OPERATOR(carlson_elliptic_integral_r_j, x, y, z, p)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_g, x, y, z)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_e, x, y, z)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(carlson_elliptic_integral_r_m, x, y, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(carlson_elliptic_integral_r_k, x, y)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_k, m)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(complete_legendre_elliptic_integral_e, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_e, phi, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(incomplete_legendre_elliptic_integral_f, phi, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(complete_legendre_elliptic_integral_pi, n, m)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(incomplete_legendre_elliptic_integral_pi, n, phi, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_amplitude_am, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dn, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cn, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sn, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sd, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cd, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_sc, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nd, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_nc, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ns, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_dc, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_ds, u, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(jacobi_elliptic_cs, u, m)

// Inverse Jacobi elliptic functions (primary)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sn, x, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cn, x, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_dn, x, m)

// Inverse Jacobi elliptic functions (derived)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sd, x, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_cd, x, m)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(inverse_jacobi_elliptic_sc, x, m)

// Jacobi theta functions
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(theta_1, z, q)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(theta_2, z, q)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(theta_3, z, q)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(theta_4, z, q)

TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_j_1, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(spherical_bessel_j, n, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_y_1, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(spherical_bessel_y, n, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_i_1, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(spherical_bessel_i, n, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_0, z)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(spherical_bessel_k_1, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(spherical_bessel_k, n, z)

// Exponential integrals
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(exponential_integral_ei, x)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(exponential_integral_e_1, x)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(exponential_integral_ein, x)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(exponential_integral_e, n, x)

// Sine integral
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(sine_integral_si, x)

// Cosine integral
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(cosine_integral_ci, x)

// Spherical Hankel functions of the first kind
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(spherical_hankel_1, n, z)

// Spherical Hankel functions of the second kind
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(spherical_hankel_2, n, z)

// Airy function of the first kind
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(airy_ai, x)

// Airy function of the second kind
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(airy_bi, x)

// Lambert W function (product logarithm)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(lambert_w, k, z)

// Kelvin function ber (real part of J_0 at rotated argument)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(kelvin_ber, x)

// Kelvin function bei (imaginary part of J_0 at rotated argument)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(kelvin_bei, x)

// Kelvin function ker (real part of K_0 at rotated argument)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(kelvin_ker, x)

// Kelvin function kei (imaginary part of K_0 at rotated argument)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(kelvin_kei, x)

// Riemann zeta function (s > 1 only)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(zeta, s)

// Polylogarithm function Li_s(z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(polylogarithm_li, s, z)

TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_u, a, z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(parabolic_cylinder_v, a, z)

// Whittaker functions
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(whittaker_m, kappa, mu, z)
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(whittaker_w, kappa, mu, z)

// Hypergeometric 0F1 (confluent hypergeometric limit function)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(hypergeometric_0_f_1, b, z)

// Hypergeometric 1F2
TORCHSCIENCE_META_POINTWISE_QUATERNARY_OPERATOR(hypergeometric_1_f_2, a, b1, b2, z)

// Faddeeva function w(z) = exp(-z^2) * erfc(-iz)
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(faddeeva_w, z)

// Inverse error function
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(erfinv, x)

// Inverse complementary error function
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(erfcinv, x)

// Fresnel sine integral
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(fresnel_s, z)

// Fresnel cosine integral
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(fresnel_c, z)

// Dawson's integral
TORCHSCIENCE_META_POINTWISE_UNARY_OPERATOR(dawson, z)

// Voigt profile
TORCHSCIENCE_META_POINTWISE_TERNARY_OPERATOR(voigt_profile, x, sigma, gamma)

// Legendre polynomial P_n(z)
TORCHSCIENCE_META_POINTWISE_BINARY_OPERATOR(legendre_polynomial_p, n, z)
