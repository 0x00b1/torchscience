#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "complete_legendre_elliptic_integral_pi.h"
#include "complete_legendre_elliptic_integral_pi_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> complete_legendre_elliptic_integral_pi_backward_backward(
    T gg_n,
    T gg_m,
    T gradient,
    T n,
    T m
) {
    // Numerical approximation for second derivatives
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());

    auto [dn, dm] = complete_legendre_elliptic_integral_pi_backward(T(1), n, m);

    // Gradient w.r.t. incoming gradient
    T grad_gradient = gg_n * dn + gg_m * dm;

    // Second derivatives via finite differences
    auto [dn_pn, dm_pn] = complete_legendre_elliptic_integral_pi_backward(T(1), n + eps, m);
    auto [dn_pm, dm_pm] = complete_legendre_elliptic_integral_pi_backward(T(1), n, m + eps);

    T d2nn = (dn_pn - dn) / eps;
    T d2nm = (dn_pm - dn) / eps;
    T d2mn = (dm_pn - dm) / eps;
    T d2mm = (dm_pm - dm) / eps;

    T grad_n = gradient * (gg_n * d2nn + gg_m * d2mn);
    T grad_m = gradient * (gg_n * d2nm + gg_m * d2mm);

    return {grad_gradient, grad_n, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
complete_legendre_elliptic_integral_pi_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_m,
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> m
) {
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> eps_c(eps, T(0));

    auto [dn, dm] = complete_legendre_elliptic_integral_pi_backward(one, n, m);

    c10::complex<T> grad_gradient = gg_n * dn + gg_m * dm;

    auto [dn_pn, dm_pn] = complete_legendre_elliptic_integral_pi_backward(one, n + eps_c, m);
    auto [dn_pm, dm_pm] = complete_legendre_elliptic_integral_pi_backward(one, n, m + eps_c);

    c10::complex<T> d2nn = (dn_pn - dn) / eps;
    c10::complex<T> d2nm = (dn_pm - dn) / eps;
    c10::complex<T> d2mn = (dm_pn - dm) / eps;
    c10::complex<T> d2mm = (dm_pm - dm) / eps;

    c10::complex<T> grad_n = gradient * (gg_n * d2nn + gg_m * d2mn);
    c10::complex<T> grad_m = gradient * (gg_n * d2nm + gg_m * d2mm);

    return {grad_gradient, grad_n, grad_m};
}

} // namespace torchscience::kernel::special_functions
