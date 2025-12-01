#pragma once

#include <tuple>
#include <cmath>

#include "jacobi_amplitude_am.h"
#include "jacobi_amplitude_am_backward.h"

namespace torchscience::kernel::special_functions {

/**
 * Second-order backward pass for Jacobi amplitude function.
 *
 * Computes the gradient of the backward pass outputs with respect to inputs.
 * Uses numerical finite differences for stability.
 *
 * Let f(u, m) = am(u, m)
 * Backward gives: (df/du, df/dm) = (dn(u, m), dam/dm)
 *
 * For second-order:
 * - grad_grad_output: derivative of loss w.r.t. forward output
 * - grad_u: accumulated gradient for u
 * - grad_m: accumulated gradient for m
 *
 * @param gradient_gradient_u Upstream gradient for df/du
 * @param gradient_gradient_m Upstream gradient for df/dm
 * @param gradient The original upstream gradient from first backward
 * @param u The argument
 * @param m The parameter
 * @return Tuple of (grad_grad_output, grad_u, grad_m)
 */
template <typename T>
std::tuple<T, T, T> jacobi_amplitude_am_backward_backward(
    T gradient_gradient_u,
    T gradient_gradient_m,
    T gradient,
    T u,
    T m
) {
    const T eps = T(1e-6);

    // Compute forward value and first derivatives
    T phi = jacobi_amplitude_am(u, m);
    T sin_phi = std::sin(phi);
    T cos_phi = std::cos(phi);
    T sin2_phi = sin_phi * sin_phi;

    // dn = sqrt(1 - m * sin^2(phi))
    T one_minus_m_sin2 = std::max(T(1) - m * sin2_phi, T(0));
    T dn = std::sqrt(one_minus_m_sin2);

    // Compute second derivatives using finite differences
    // d^2(am)/du^2, d^2(am)/dm^2, d^2(am)/dudm

    // For grad_u in backward: grad * dn
    // d(grad * dn)/du = grad * d(dn)/du
    // d(grad * dn)/dm = grad * d(dn)/dm
    // d(grad * dn)/d(grad) = dn

    // For grad_m in backward: grad * dam/dm (numerical)
    // Similar analysis

    // Use numerical approach for second derivatives
    auto backward_func = [](T u_val, T m_val) -> std::tuple<T, T> {
        T phi_val = jacobi_amplitude_am(u_val, m_val);
        T sin_val = std::sin(phi_val);
        T dn_val = std::sqrt(std::max(T(1) - m_val * sin_val * sin_val, T(0)));

        // Numerical gradient w.r.t. m
        T eps_local = T(1e-7);
        T phi_plus, phi_minus, grad_m_val;
        if (m_val < eps_local) {
            phi_plus = jacobi_amplitude_am(u_val, m_val + eps_local);
            grad_m_val = (phi_plus - phi_val) / eps_local;
        } else if (m_val > T(1) - eps_local) {
            phi_minus = jacobi_amplitude_am(u_val, m_val - eps_local);
            grad_m_val = (phi_val - phi_minus) / eps_local;
        } else {
            phi_plus = jacobi_amplitude_am(u_val, m_val + eps_local);
            phi_minus = jacobi_amplitude_am(u_val, m_val - eps_local);
            grad_m_val = (phi_plus - phi_minus) / (T(2) * eps_local);
        }

        return {dn_val, grad_m_val};
    };

    // Current values
    auto [dn_curr, dam_dm_curr] = backward_func(u, m);

    // Finite differences for second-order terms
    // d(dn)/du and d(dam_dm)/du
    auto [dn_u_plus, dam_dm_u_plus] = backward_func(u + eps, m);
    auto [dn_u_minus, dam_dm_u_minus] = backward_func(u - eps, m);
    T d_dn_du = (dn_u_plus - dn_u_minus) / (T(2) * eps);
    T d_dam_dm_du = (dam_dm_u_plus - dam_dm_u_minus) / (T(2) * eps);

    // d(dn)/dm and d(dam_dm)/dm
    T d_dn_dm, d_dam_dm_dm;
    if (m < eps) {
        auto [dn_m_plus, dam_dm_m_plus] = backward_func(u, m + eps);
        d_dn_dm = (dn_m_plus - dn_curr) / eps;
        d_dam_dm_dm = (dam_dm_m_plus - dam_dm_curr) / eps;
    } else if (m > T(1) - eps) {
        auto [dn_m_minus, dam_dm_m_minus] = backward_func(u, m - eps);
        d_dn_dm = (dn_curr - dn_m_minus) / eps;
        d_dam_dm_dm = (dam_dm_curr - dam_dm_m_minus) / eps;
    } else {
        auto [dn_m_plus, dam_dm_m_plus] = backward_func(u, m + eps);
        auto [dn_m_minus, dam_dm_m_minus] = backward_func(u, m - eps);
        d_dn_dm = (dn_m_plus - dn_m_minus) / (T(2) * eps);
        d_dam_dm_dm = (dam_dm_m_plus - dam_dm_m_minus) / (T(2) * eps);
    }

    // Compute outputs:
    // grad_grad_output = gradient_gradient_u * dn + gradient_gradient_m * dam_dm
    T grad_grad_output = gradient_gradient_u * dn_curr + gradient_gradient_m * dam_dm_curr;

    // grad_u = gradient * (gradient_gradient_u * d_dn_du + gradient_gradient_m * d_dam_dm_du)
    T grad_u = gradient * (gradient_gradient_u * d_dn_du + gradient_gradient_m * d_dam_dm_du);

    // grad_m = gradient * (gradient_gradient_u * d_dn_dm + gradient_gradient_m * d_dam_dm_dm)
    T grad_m = gradient * (gradient_gradient_u * d_dn_dm + gradient_gradient_m * d_dam_dm_dm);

    return {grad_grad_output, grad_u, grad_m};
}

/**
 * Complex version of second-order backward pass.
 */
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
jacobi_amplitude_am_backward_backward(
    c10::complex<T> gradient_gradient_u,
    c10::complex<T> gradient_gradient_m,
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    const T eps = T(1e-6);
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> eps_c(eps, T(0));

    auto backward_func = [](c10::complex<T> u_val, c10::complex<T> m_val)
        -> std::tuple<c10::complex<T>, c10::complex<T>> {
        c10::complex<T> phi_val = jacobi_amplitude_am(u_val, m_val);
        c10::complex<T> sin_val = std::sin(phi_val);
        c10::complex<T> one_c(T(1), T(0));
        c10::complex<T> dn_val = std::sqrt(one_c - m_val * sin_val * sin_val);

        // Numerical gradient w.r.t. m
        T eps_local = T(1e-7);
        c10::complex<T> eps_local_c(eps_local, T(0));
        c10::complex<T> two_c(T(2), T(0));
        c10::complex<T> phi_plus = jacobi_amplitude_am(u_val, m_val + eps_local_c);
        c10::complex<T> phi_minus = jacobi_amplitude_am(u_val, m_val - eps_local_c);
        c10::complex<T> grad_m_val = (phi_plus - phi_minus) / (two_c * eps_local_c);

        return {dn_val, grad_m_val};
    };

    // Current values
    auto [dn_curr, dam_dm_curr] = backward_func(u, m);

    // Finite differences for second-order terms
    auto [dn_u_plus, dam_dm_u_plus] = backward_func(u + eps_c, m);
    auto [dn_u_minus, dam_dm_u_minus] = backward_func(u - eps_c, m);
    c10::complex<T> d_dn_du = (dn_u_plus - dn_u_minus) / (two * eps_c);
    c10::complex<T> d_dam_dm_du = (dam_dm_u_plus - dam_dm_u_minus) / (two * eps_c);

    auto [dn_m_plus, dam_dm_m_plus] = backward_func(u, m + eps_c);
    auto [dn_m_minus, dam_dm_m_minus] = backward_func(u, m - eps_c);
    c10::complex<T> d_dn_dm = (dn_m_plus - dn_m_minus) / (two * eps_c);
    c10::complex<T> d_dam_dm_dm = (dam_dm_m_plus - dam_dm_m_minus) / (two * eps_c);

    // Compute outputs
    c10::complex<T> grad_grad_output = gradient_gradient_u * dn_curr + gradient_gradient_m * dam_dm_curr;
    c10::complex<T> grad_u = gradient * (gradient_gradient_u * d_dn_du + gradient_gradient_m * d_dam_dm_du);
    c10::complex<T> grad_m = gradient * (gradient_gradient_u * d_dn_dm + gradient_gradient_m * d_dam_dm_dm);

    return {grad_grad_output, grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
