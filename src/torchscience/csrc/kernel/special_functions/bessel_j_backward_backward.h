#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "bessel_j.h"
#include "bessel_j_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d²Jₙ/dz²
// Using: d/dz[(J_{n-1} - J_{n+1})/2] = (J_{n-2} - 2Jₙ + J_{n+2})/4
// Or equivalently from Bessel's equation: d²Jₙ/dz² = -Jₙ/z + n²Jₙ/z² - dJₙ/dz/z
// We use the recurrence approach for clarity
template <typename T>
T bessel_j_zz_derivative(T n, T z) {
    // d²/dz² Jₙ(z) = (d/dz)[d/dz Jₙ(z)]
    // d/dz Jₙ(z) = (Jₙ₋₁ - Jₙ₊₁)/2
    // d/dz[(Jₙ₋₁ - Jₙ₊₁)/2] = [(Jₙ₋₂ - Jₙ)/2 - (Jₙ - Jₙ₊₂)/2]/2
    //                        = (Jₙ₋₂ - 2Jₙ + Jₙ₊₂)/4
    T j_nm2 = bessel_j(n - T(2), z);
    T j_n = bessel_j(n, z);
    T j_np2 = bessel_j(n + T(2), z);

    return (j_nm2 - T(2) * j_n + j_np2) / T(4);
}

// Mixed second derivative ∂²Jₙ/∂n∂z computed numerically
template <typename T>
T bessel_j_nz_derivative(T n, T z) {
    const T eps = std::sqrt(bessel_j_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // d/dz J_{n+h} and d/dz J_{n-h}
    T j_p_nm1 = bessel_j(n + h - T(1), z);
    T j_p_np1 = bessel_j(n + h + T(1), z);
    T dj_dz_plus = (j_p_nm1 - j_p_np1) / T(2);

    T j_m_nm1 = bessel_j(n - h - T(1), z);
    T j_m_np1 = bessel_j(n - h + T(1), z);
    T dj_dz_minus = (j_m_nm1 - j_m_np1) / T(2);

    return (dj_dz_plus - dj_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: ∂²Jₙ/∂n² computed numerically
template <typename T>
T bessel_j_nn_derivative(T n, T z) {
    const T eps = std::cbrt(bessel_j_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T j_plus = bessel_j(n + h, z);
    T j_center = bessel_j(n, z);
    T j_minus = bessel_j(n - h, z);

    return (j_plus - T(2) * j_center + j_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> bessel_j_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    c10::complex<T> j_nm2 = bessel_j(n - two, z);
    c10::complex<T> j_n = bessel_j(n, z);
    c10::complex<T> j_np2 = bessel_j(n + two, z);

    return (j_nm2 - two * j_n + j_np2) / four;
}

template <typename T>
c10::complex<T> bessel_j_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(bessel_j_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> j_p_nm1 = bessel_j(n + h - one, z);
    c10::complex<T> j_p_np1 = bessel_j(n + h + one, z);
    c10::complex<T> dj_dz_plus = (j_p_nm1 - j_p_np1) / two;

    c10::complex<T> j_m_nm1 = bessel_j(n - h - one, z);
    c10::complex<T> j_m_np1 = bessel_j(n - h + one, z);
    c10::complex<T> dj_dz_minus = (j_m_nm1 - j_m_np1) / two;

    return (dj_dz_plus - dj_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> bessel_j_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(bessel_j_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> j_plus = bessel_j(n + h, z);
    c10::complex<T> j_center = bessel_j(n, z);
    c10::complex<T> j_minus = bessel_j(n - h, z);

    return (j_plus - two * j_center + j_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> bessel_j_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    // Forward backward computes:
    // grad_n = grad_output * dJ/dn
    // grad_z = grad_output * (J_{n-1} - J_{n+1})/2

    // We need:
    // d(grad_n)/d(grad_output) = dJ/dn
    // d(grad_n)/dn = grad_output * d²J/dn²
    // d(grad_n)/dz = grad_output * d²J/dndz

    // d(grad_z)/d(grad_output) = (J_{n-1} - J_{n+1})/2
    // d(grad_z)/dn = grad_output * d/dn[(J_{n-1} - J_{n+1})/2]
    // d(grad_z)/dz = grad_output * d²J/dz² = grad_output * (J_{n-2} - 2J_n + J_{n+2})/4

    // First derivatives
    T j_nm1 = bessel_j(n - T(1), z);
    T j_np1 = bessel_j(n + T(1), z);
    T dj_dz = (j_nm1 - j_np1) / T(2);

    T dj_dn = detail::bessel_j_n_derivative(n, z);

    // Second derivatives
    T d2j_dz2 = detail::bessel_j_zz_derivative(n, z);
    T d2j_dn2 = detail::bessel_j_nn_derivative(n, z);
    T d2j_dndz = detail::bessel_j_nz_derivative(n, z);

    // d(grad_z)/dn: need d/dn[(J_{n-1} - J_{n+1})/2]
    // This equals (dJ_{n-1}/dn - dJ_{n+1}/dn)/2
    // But dJ_{n-1}/dn = dJ_{n-1}/d(n-1) since J_{n-1} depends on (n-1)
    // This is complex; we approximate numerically
    T d_dz_dn = d2j_dndz;  // d²J/dndz = d/dn[dJ/dz]

    // Accumulate gradients
    // grad_grad_output = gg_n * dJ/dn + gg_z * dJ/dz
    T grad_grad_output = gg_n * dj_dn + gg_z * dj_dz;

    // grad_n = gg_n * grad_output * d²J/dn² + gg_z * grad_output * d²J/dndz
    T grad_n = grad_output * (gg_n * d2j_dn2 + gg_z * d_dz_dn);

    // grad_z = gg_n * grad_output * d²J/dndz + gg_z * grad_output * d²J/dz²
    T grad_z = grad_output * (gg_n * d2j_dndz + gg_z * d2j_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> bessel_j_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // First derivatives
    c10::complex<T> j_nm1 = bessel_j(n - one, z);
    c10::complex<T> j_np1 = bessel_j(n + one, z);
    c10::complex<T> dj_dz = (j_nm1 - j_np1) / two;

    c10::complex<T> dj_dn = detail::bessel_j_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2j_dz2 = detail::bessel_j_zz_derivative(n, z);
    c10::complex<T> d2j_dn2 = detail::bessel_j_nn_derivative(n, z);
    c10::complex<T> d2j_dndz = detail::bessel_j_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(dj_dn) + gg_z * std::conj(dj_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2j_dn2) + gg_z * std::conj(d2j_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2j_dndz) + gg_z * std::conj(d2j_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
