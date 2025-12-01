#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "incomplete_legendre_elliptic_integral_pi.h"
#include "incomplete_legendre_elliptic_integral_pi_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> incomplete_legendre_elliptic_integral_pi_backward_backward(
    T gg_n,
    T gg_phi,
    T gg_m,
    T gradient,
    T n,
    T phi,
    T m
) {
    // Numerical approximation for second derivatives
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());

    auto [dn, dphi, dm] = incomplete_legendre_elliptic_integral_pi_backward(T(1), n, phi, m);

    // Gradient w.r.t. incoming gradient
    T grad_gradient = gg_n * dn + gg_phi * dphi + gg_m * dm;

    // Second derivatives via finite differences
    auto [dn_pn, dphi_pn, dm_pn] = incomplete_legendre_elliptic_integral_pi_backward(T(1), n + eps, phi, m);
    auto [dn_pphi, dphi_pphi, dm_pphi] = incomplete_legendre_elliptic_integral_pi_backward(T(1), n, phi + eps, m);
    auto [dn_pm, dphi_pm, dm_pm] = incomplete_legendre_elliptic_integral_pi_backward(T(1), n, phi, m + eps);

    // Hessian elements (computed via forward differences)
    T d2nn = (dn_pn - dn) / eps;
    T d2nphi = (dn_pphi - dn) / eps;
    T d2nm = (dn_pm - dn) / eps;

    T d2phin = (dphi_pn - dphi) / eps;
    T d2phiphi = (dphi_pphi - dphi) / eps;
    T d2phim = (dphi_pm - dphi) / eps;

    T d2mn = (dm_pn - dm) / eps;
    T d2mphi = (dm_pphi - dm) / eps;
    T d2mm = (dm_pm - dm) / eps;

    // Chain rule for second-order gradients
    T grad_n = gradient * (gg_n * d2nn + gg_phi * d2phin + gg_m * d2mn);
    T grad_phi = gradient * (gg_n * d2nphi + gg_phi * d2phiphi + gg_m * d2mphi);
    T grad_m = gradient * (gg_n * d2nm + gg_phi * d2phim + gg_m * d2mm);

    return {grad_gradient, grad_n, grad_phi, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
incomplete_legendre_elliptic_integral_pi_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_phi,
    c10::complex<T> gg_m,
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> phi,
    c10::complex<T> m
) {
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> eps_c(eps, T(0));

    auto [dn, dphi, dm] = incomplete_legendre_elliptic_integral_pi_backward(one, n, phi, m);

    c10::complex<T> grad_gradient = gg_n * dn + gg_phi * dphi + gg_m * dm;

    auto [dn_pn, dphi_pn, dm_pn] = incomplete_legendre_elliptic_integral_pi_backward(one, n + eps_c, phi, m);
    auto [dn_pphi, dphi_pphi, dm_pphi] = incomplete_legendre_elliptic_integral_pi_backward(one, n, phi + eps_c, m);
    auto [dn_pm, dphi_pm, dm_pm] = incomplete_legendre_elliptic_integral_pi_backward(one, n, phi, m + eps_c);

    c10::complex<T> d2nn = (dn_pn - dn) / eps;
    c10::complex<T> d2nphi = (dn_pphi - dn) / eps;
    c10::complex<T> d2nm = (dn_pm - dn) / eps;

    c10::complex<T> d2phin = (dphi_pn - dphi) / eps;
    c10::complex<T> d2phiphi = (dphi_pphi - dphi) / eps;
    c10::complex<T> d2phim = (dphi_pm - dphi) / eps;

    c10::complex<T> d2mn = (dm_pn - dm) / eps;
    c10::complex<T> d2mphi = (dm_pphi - dm) / eps;
    c10::complex<T> d2mm = (dm_pm - dm) / eps;

    c10::complex<T> grad_n = gradient * (gg_n * d2nn + gg_phi * d2phin + gg_m * d2mn);
    c10::complex<T> grad_phi = gradient * (gg_n * d2nphi + gg_phi * d2phiphi + gg_m * d2mphi);
    c10::complex<T> grad_m = gradient * (gg_n * d2nm + gg_phi * d2phim + gg_m * d2mm);

    return {grad_gradient, grad_n, grad_phi, grad_m};
}

} // namespace torchscience::kernel::special_functions
