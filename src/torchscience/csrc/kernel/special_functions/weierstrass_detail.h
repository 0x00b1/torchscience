#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::special_functions::weierstrass_detail {

// ============================================================================
// Tolerance templates for numerical convergence
// ============================================================================

template <typename T>
inline T tolerance() {
    return T(1e-10);
}

template <>
inline float tolerance<float>() { return 1e-6f; }

template <>
inline double tolerance<double>() { return 1e-14; }

template <>
inline c10::Half tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

// Maximum iterations for iterative algorithms
constexpr int max_iterations = 100;

// ============================================================================
// cbrt_real: Cube root for real numbers (handles negative values)
// ============================================================================

// Standard cbrt doesn't handle negative real inputs correctly in all contexts.
// This function returns the real cube root for negative real inputs.
template <typename T>
T cbrt_real(T x) {
    if (x >= T(0)) {
        return std::cbrt(x);
    } else {
        return -std::cbrt(-x);
    }
}

// ============================================================================
// solve_depressed_cubic: Solve t^3 + pt + q = 0 using Cardano's formula
// Returns 3 complex roots
// ============================================================================

// Cardano's formula for depressed cubic t^3 + pt + q = 0:
// Let discriminant D = (q/2)^2 + (p/3)^3
//
// Roots are: t_k = omega^k * u + omega^{2k} * v, for k = 0, 1, 2
// where omega = e^{2*pi*i/3} is a primitive cube root of unity,
// and u, v are the cube roots satisfying u^3 = -q/2 + sqrt(D), v^3 = -q/2 - sqrt(D)
// with the constraint uv = -p/3

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
solve_depressed_cubic(T p, T q) {
    const T tol = tolerance<T>();

    // Primitive cube root of unity: omega = e^{2*pi*i/3} = -1/2 + i*sqrt(3)/2
    const c10::complex<T> omega(T(-0.5), std::sqrt(T(3)) / T(2));
    const c10::complex<T> omega_sq(T(-0.5), -std::sqrt(T(3)) / T(2));

    // Handle special case p = 0: t^3 + q = 0 => t = (-q)^{1/3}
    if (std::abs(p) < tol) {
        c10::complex<T> base(-q, T(0));
        // Principal cube root
        T r = std::pow(std::abs(base), T(1) / T(3));
        T theta = std::arg(base) / T(3);
        c10::complex<T> t0(r * std::cos(theta), r * std::sin(theta));
        c10::complex<T> t1 = t0 * omega;
        c10::complex<T> t2 = t0 * omega_sq;
        return std::make_tuple(t0, t1, t2);
    }

    // Discriminant D = (q/2)^2 + (p/3)^3
    T q_half = q / T(2);
    T p_third = p / T(3);
    T D = q_half * q_half + p_third * p_third * p_third;

    c10::complex<T> sqrt_D;
    if (D >= T(0)) {
        sqrt_D = c10::complex<T>(std::sqrt(D), T(0));
    } else {
        sqrt_D = c10::complex<T>(T(0), std::sqrt(-D));
    }

    // u^3 = -q/2 + sqrt(D), v^3 = -q/2 - sqrt(D)
    c10::complex<T> u_cubed = c10::complex<T>(-q_half, T(0)) + sqrt_D;
    c10::complex<T> v_cubed = c10::complex<T>(-q_half, T(0)) - sqrt_D;

    // Compute principal cube roots
    T r_u = std::pow(std::abs(u_cubed), T(1) / T(3));
    T theta_u = std::arg(u_cubed) / T(3);
    c10::complex<T> u(r_u * std::cos(theta_u), r_u * std::sin(theta_u));

    T r_v = std::pow(std::abs(v_cubed), T(1) / T(3));
    T theta_v = std::arg(v_cubed) / T(3);
    c10::complex<T> v(r_v * std::cos(theta_v), r_v * std::sin(theta_v));

    // We need uv = -p/3. Check and adjust v if needed.
    // If uv != -p/3, multiply v by omega or omega^2
    c10::complex<T> target(-p_third, T(0));
    c10::complex<T> uv = u * v;

    if (std::abs(uv - target) > tol) {
        c10::complex<T> v1 = v * omega;
        if (std::abs(u * v1 - target) < std::abs(uv - target)) {
            v = v1;
        } else {
            c10::complex<T> v2 = v * omega_sq;
            if (std::abs(u * v2 - target) < std::abs(uv - target)) {
                v = v2;
            }
        }
    }

    // The three roots: t_k = omega^k * u + omega^{2k} * v
    c10::complex<T> t0 = u + v;
    c10::complex<T> t1 = omega * u + omega_sq * v;
    c10::complex<T> t2 = omega_sq * u + omega * v;

    return std::make_tuple(t0, t1, t2);
}

// Complex version of solve_depressed_cubic
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
solve_depressed_cubic(c10::complex<T> p, c10::complex<T> q) {
    const T tol = tolerance<T>();

    // Primitive cube root of unity
    const c10::complex<T> omega(T(-0.5), std::sqrt(T(3)) / T(2));
    const c10::complex<T> omega_sq(T(-0.5), -std::sqrt(T(3)) / T(2));
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> three(T(3), T(0));

    // Handle special case p = 0
    if (std::abs(p) < tol) {
        c10::complex<T> base = -q;
        T r = std::pow(std::abs(base), T(1) / T(3));
        T theta = std::arg(base) / T(3);
        c10::complex<T> t0(r * std::cos(theta), r * std::sin(theta));
        c10::complex<T> t1 = t0 * omega;
        c10::complex<T> t2 = t0 * omega_sq;
        return std::make_tuple(t0, t1, t2);
    }

    // Discriminant D = (q/2)^2 + (p/3)^3
    c10::complex<T> q_half = q / two;
    c10::complex<T> p_third = p / three;
    c10::complex<T> D = q_half * q_half + p_third * p_third * p_third;

    c10::complex<T> sqrt_D = std::sqrt(D);

    // u^3 = -q/2 + sqrt(D), v^3 = -q/2 - sqrt(D)
    c10::complex<T> u_cubed = -q_half + sqrt_D;
    c10::complex<T> v_cubed = -q_half - sqrt_D;

    // Compute principal cube roots
    T r_u = std::pow(std::abs(u_cubed), T(1) / T(3));
    T theta_u = std::arg(u_cubed) / T(3);
    c10::complex<T> u(r_u * std::cos(theta_u), r_u * std::sin(theta_u));

    T r_v = std::pow(std::abs(v_cubed), T(1) / T(3));
    T theta_v = std::arg(v_cubed) / T(3);
    c10::complex<T> v(r_v * std::cos(theta_v), r_v * std::sin(theta_v));

    // Adjust v so that uv = -p/3
    c10::complex<T> target = -p_third;
    c10::complex<T> uv = u * v;

    if (std::abs(uv - target) > tol) {
        c10::complex<T> v1 = v * omega;
        if (std::abs(u * v1 - target) < std::abs(uv - target)) {
            v = v1;
        } else {
            c10::complex<T> v2 = v * omega_sq;
            if (std::abs(u * v2 - target) < std::abs(uv - target)) {
                v = v2;
            }
        }
    }

    c10::complex<T> t0 = u + v;
    c10::complex<T> t1 = omega * u + omega_sq * v;
    c10::complex<T> t2 = omega_sq * u + omega * v;

    return std::make_tuple(t0, t1, t2);
}

// ============================================================================
// invariants_to_roots: Find roots e1, e2, e3 of 4t^3 - g2*t - g3 = 0
// ============================================================================

// The Weierstrass elliptic function satisfies (P')^2 = 4P^3 - g2*P - g3
// The roots e1, e2, e3 are the values of P at the half-periods.
// We solve 4t^3 - g2*t - g3 = 0 by converting to depressed cubic:
// t^3 - (g2/4)*t - (g3/4) = 0

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
invariants_to_roots(T g2, T g3) {
    // Convert 4t^3 - g2*t - g3 = 0 to depressed cubic t^3 + pt + q = 0
    // Dividing by 4: t^3 - (g2/4)*t - (g3/4) = 0
    // So p = -g2/4, q = -g3/4
    T p = -g2 / T(4);
    T q = -g3 / T(4);

    return solve_depressed_cubic(p, q);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
invariants_to_roots(c10::complex<T> g2, c10::complex<T> g3) {
    c10::complex<T> four(T(4), T(0));
    c10::complex<T> p = -g2 / four;
    c10::complex<T> q = -g3 / four;

    return solve_depressed_cubic(p, q);
}

// ============================================================================
// roots_to_half_periods: Compute half-periods omega1, omega3 using AGM
// ============================================================================

// Given the roots e1, e2, e3 (with e1 + e2 + e3 = 0), compute half-periods
// using the arithmetic-geometric mean (AGM) method.
//
// The half-periods are related to complete elliptic integrals:
// omega1 = K(m) / sqrt(e1 - e3)
// omega3 = i * K(1-m) / sqrt(e1 - e3)
// where m = (e2 - e3) / (e1 - e3)
//
// We use the AGM method: K(m) = pi / (2 * agm(1, sqrt(1-m)))

template <typename T>
c10::complex<T> agm(c10::complex<T> a, c10::complex<T> b) {
    const T tol = tolerance<T>();

    for (int i = 0; i < max_iterations; ++i) {
        c10::complex<T> a_new = (a + b) / T(2);
        c10::complex<T> b_new = std::sqrt(a * b);

        // Choose the sign of b_new to minimize |a_new - b_new|
        // (AGM requires consistent branch choice)
        if (std::abs(a_new - b_new) > std::abs(a_new + b_new)) {
            b_new = -b_new;
        }

        a = a_new;
        b = b_new;

        if (std::abs(a - b) < tol * std::abs(a)) {
            break;
        }
    }

    return a;
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>>
roots_to_half_periods(c10::complex<T> e1, c10::complex<T> e2, c10::complex<T> e3) {
    const T pi = T(3.14159265358979323846);
    const c10::complex<T> I(T(0), T(1));
    const c10::complex<T> one(T(1), T(0));

    // Compute differences
    c10::complex<T> d12 = e1 - e2;
    c10::complex<T> d13 = e1 - e3;
    c10::complex<T> d23 = e2 - e3;

    // Elliptic parameter m = (e2 - e3) / (e1 - e3)
    c10::complex<T> m = d23 / d13;
    c10::complex<T> m1 = one - m;  // 1 - m = (e1 - e2) / (e1 - e3)

    // Compute complete elliptic integrals via AGM
    // K(m) = pi / (2 * agm(1, sqrt(1-m)))
    c10::complex<T> agm_K = agm(one, std::sqrt(m1));
    c10::complex<T> K_m = c10::complex<T>(pi / T(2), T(0)) / agm_K;

    // K'(m) = K(1-m) = pi / (2 * agm(1, sqrt(m)))
    c10::complex<T> agm_K_prime = agm(one, std::sqrt(m));
    c10::complex<T> K_prime = c10::complex<T>(pi / T(2), T(0)) / agm_K_prime;

    // Half-periods
    c10::complex<T> sqrt_d13 = std::sqrt(d13);
    c10::complex<T> omega1 = K_m / sqrt_d13;
    c10::complex<T> omega3 = I * K_prime / sqrt_d13;

    return std::make_tuple(omega1, omega3);
}

// Real version - converts to complex and back
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>>
roots_to_half_periods(T e1, T e2, T e3) {
    return roots_to_half_periods(
        c10::complex<T>(e1, T(0)),
        c10::complex<T>(e2, T(0)),
        c10::complex<T>(e3, T(0))
    );
}

// ============================================================================
// half_periods_to_nome: Compute q = exp(i*pi*tau) where tau = omega3/omega1
// ============================================================================

// The nome q is defined as q = exp(i*pi*tau) where tau = omega3/omega1
// is the period ratio. For convergence of theta series, we need |q| < 1,
// which requires Im(tau) > 0.

template <typename T>
c10::complex<T> half_periods_to_nome(c10::complex<T> omega1, c10::complex<T> omega3) {
    const T pi = T(3.14159265358979323846);
    const c10::complex<T> I(T(0), T(1));

    // Period ratio tau = omega3 / omega1
    c10::complex<T> tau = omega3 / omega1;

    // Nome q = exp(i*pi*tau)
    c10::complex<T> exponent = I * c10::complex<T>(pi, T(0)) * tau;
    return std::exp(exponent);
}

template <typename T>
c10::complex<T> half_periods_to_nome(T omega1_real, T omega1_imag, T omega3_real, T omega3_imag) {
    c10::complex<T> omega1(omega1_real, omega1_imag);
    c10::complex<T> omega3(omega3_real, omega3_imag);
    return half_periods_to_nome(omega1, omega3);
}

// ============================================================================
// invariants_to_lattice_params: Combined function for full conversion
// ============================================================================

// Converts Weierstrass invariants (g2, g3) to lattice parameters (omega1, omega3, q)
// This combines: invariants -> roots -> half_periods -> nome

template <typename T>
struct LatticeParams {
    c10::complex<T> omega1;  // First half-period (typically real for real g2, g3 with Delta > 0)
    c10::complex<T> omega3;  // Third half-period (typically purely imaginary)
    c10::complex<T> q;       // Nome
    c10::complex<T> e1;      // Root e1
    c10::complex<T> e2;      // Root e2
    c10::complex<T> e3;      // Root e3
};

template <typename T>
LatticeParams<T> invariants_to_lattice_params(T g2, T g3) {
    LatticeParams<T> result;

    // Find roots of 4t^3 - g2*t - g3 = 0
    auto [e1, e2, e3] = invariants_to_roots(g2, g3);

    // Order roots: for real discriminant > 0, we want e1 > e2 > e3 (all real)
    // For complex roots, ordering is by convention
    // Here we just use the order returned by Cardano's formula
    result.e1 = e1;
    result.e2 = e2;
    result.e3 = e3;

    // Compute half-periods
    auto [omega1, omega3] = roots_to_half_periods(e1, e2, e3);
    result.omega1 = omega1;
    result.omega3 = omega3;

    // Compute nome
    result.q = half_periods_to_nome(omega1, omega3);

    return result;
}

template <typename T>
LatticeParams<T> invariants_to_lattice_params(c10::complex<T> g2, c10::complex<T> g3) {
    LatticeParams<T> result;

    // Find roots
    auto [e1, e2, e3] = invariants_to_roots(g2, g3);
    result.e1 = e1;
    result.e2 = e2;
    result.e3 = e3;

    // Compute half-periods
    auto [omega1, omega3] = roots_to_half_periods(e1, e2, e3);
    result.omega1 = omega1;
    result.omega3 = omega3;

    // Compute nome
    result.q = half_periods_to_nome(omega1, omega3);

    return result;
}

// ============================================================================
// Utility: Modular discriminant Delta = g2^3 - 27*g3^2
// ============================================================================

template <typename T>
T discriminant(T g2, T g3) {
    return g2 * g2 * g2 - T(27) * g3 * g3;
}

template <typename T>
c10::complex<T> discriminant(c10::complex<T> g2, c10::complex<T> g3) {
    return g2 * g2 * g2 - c10::complex<T>(T(27), T(0)) * g3 * g3;
}

} // namespace torchscience::kernel::special_functions::weierstrass_detail
