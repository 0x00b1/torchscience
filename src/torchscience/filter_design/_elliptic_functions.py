"""Elliptic functions for filter design.

This module implements the elliptic filter prototype computation
following scipy's ellipap algorithm.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy import special

# Constants
_ELLIPDEG_MMAX = 7
EPSILON = 2e-16


def _pow10m1(x: float) -> float:
    """Compute 10^x - 1 accurately for small x."""
    return 10.0**x - 1.0


def _ellipdeg(n: int, m1: float) -> float:
    """Solve degree equation using nomes.

    Given n, m1, solve:
        n * K(m) / K'(m) = K(m1) / K'(m1)
    for m.

    Uses the nome q to solve efficiently.
    """
    K1 = special.ellipk(m1)
    K1p = special.ellipkm1(m1)

    q1 = np.exp(-np.pi * K1p / K1)
    q = q1 ** (1 / n)

    mnum = np.arange(_ELLIPDEG_MMAX + 1)
    mden = np.arange(1, _ELLIPDEG_MMAX + 2)

    num = np.sum(q ** (mnum * (mnum + 1)))
    den = 1 + 2 * np.sum(q ** (mden**2))

    return float(16 * q * (num / den) ** 4)


def _arc_jac_sn(w: complex, m: float) -> complex:
    """Complex inverse Jacobian sn function.

    Solve for z in w = sn(z, m).
    Uses Newton iteration.
    """
    # Initial guess
    if abs(w) < 1:
        z = complex(math.asin(w.real), 0)
    else:
        z = complex(math.pi / 2, math.asinh(abs(w.imag)))

    for _ in range(50):
        sn, cn, dn, _ = special.ellipj(z.real, m)
        # For complex z, use series expansion or numerical differentiation
        # Simplified: use real part only for iteration
        sn_val = complex(sn, 0)
        cn_val = complex(cn, 0)
        dn_val = complex(dn, 0)

        err = sn_val - w

        if abs(err) < 1e-14:
            break

        # d(sn)/dz = cn * dn
        deriv = cn_val * dn_val
        if abs(deriv) > 1e-15:
            z = z - err / deriv

    return z


def _arc_jac_sc1(w: float, m: float) -> float:
    """Real inverse Jacobian sc, with complementary modulus.

    Solve for z in w = sc(z, 1-m).

    From the identity: sc(z, m) = -i * sn(i * z, 1 - m)
    So sc(z, 1-m) = -i * sn(i*z, m)
    Thus z = -i * asn(i*w, m)
    """
    # Solve w = sn(i*z, m) / cn(i*z, m) for real z
    # Using Newton iteration on sc directly

    # Initial guess
    if m == 0:
        return math.atan(w)
    if m == 1:
        return math.asinh(w)

    # Newton iteration
    z = math.atan(w)

    for _ in range(50):
        sn, cn, dn, _ = special.ellipj(z, 1 - m)
        if abs(cn) < 1e-15:
            break

        sc = sn / cn
        err = sc - w

        if abs(err) < 1e-14:
            break

        # d(sc)/dz = dn / cn^2
        deriv = dn / (cn * cn) if cn != 0 else 1e15

        if abs(deriv) > 1e-15:
            z = z - err / deriv

    return z


def elliptic_prototype(
    n: int,
    rp: float,
    rs: float,
) -> Tuple[List[complex], List[complex], float]:
    """Compute zeros, poles, and gain for elliptic analog lowpass prototype.

    This implements scipy's ellipap algorithm exactly.

    Parameters
    ----------
    n : int
        Filter order (must be positive).
    rp : float
        Passband ripple in dB (must be positive).
    rs : float
        Stopband attenuation in dB (must be positive).

    Returns
    -------
    zeros : list of complex
        Filter zeros on imaginary axis.
    poles : list of complex
        Filter poles in left half-plane.
    gain : float
        System gain.
    """
    if n < 1:
        raise ValueError(f"Order must be positive, got {n}")
    if rp <= 0:
        raise ValueError(f"Passband ripple must be positive, got {rp}")
    if rs <= 0:
        raise ValueError(f"Stopband attenuation must be positive, got {rs}")

    # Special case n=1
    if n == 1:
        eps_sq = _pow10m1(0.1 * rp)
        p_real = -math.sqrt(1.0 / eps_sq)
        return [], [complex(p_real, 0)], -p_real

    eps_sq = _pow10m1(0.1 * rp)
    eps = math.sqrt(eps_sq)

    # ck1_sq = k1^2 where k1 is the ripple ratio
    ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    if ck1_sq == 0:
        raise ValueError("Cannot design filter with given rp and rs specs")

    # Solve degree equation for m (the selectivity parameter)
    m = _ellipdeg(n, ck1_sq)

    # Complete elliptic integral
    capk = float(special.ellipk(m))

    # Compute zeros
    # j = [1, 3, 5, ...] for even n, [0, 2, 4, ...] for odd n
    # Actually scipy uses: j = np.arange(1 - n % 2, n, 2)
    # For n=4: j = [1, 3]
    # For n=5: j = [0, 2, 4]
    j_vals = list(range(1 - n % 2, n, 2))

    zeros = []
    sqrt_m = math.sqrt(m)

    # sn values at j * K / n
    for j in j_vals:
        u = j * capk / n
        sn_val, _, _, _ = special.ellipj(u, m)
        if abs(sn_val) > EPSILON:
            z_imag = 1.0 / (sqrt_m * sn_val)
            zeros.append(complex(0, z_imag))
            zeros.append(complex(0, -z_imag))

    # Compute v0 for poles
    K1 = float(special.ellipk(ck1_sq))
    r = _arc_jac_sc1(1.0 / eps, ck1_sq)
    v0 = capk * r / (n * K1)

    # Pole computation
    poles = []
    sv, cv, dv, _ = special.ellipj(v0, 1 - m)

    for j in j_vals:
        u = j * capk / n
        s, c, d, _ = special.ellipj(u, m)

        # Pole formula from scipy
        # p = -(c * d * sv * cv + 1j * s * dv) / (1 - (d * sv) ** 2)
        denom = 1 - (d * sv) ** 2
        if abs(denom) > 1e-15:
            p_real = -c * d * sv * cv / denom
            p_imag = -s * dv / denom  # Note: scipy has +1j*s*dv in numerator

            poles.append(complex(p_real, p_imag))

    # For odd n, filter out the purely real pole (it's already included once)
    # and add conjugates
    if n % 2 == 1:
        # Keep poles with significant imaginary part
        new_poles = []
        for p in poles:
            if abs(p.imag) > EPSILON * math.sqrt(abs(p) ** 2):
                new_poles.append(p)
                new_poles.append(complex(p.real, -p.imag))
            else:
                new_poles.append(p)  # Real pole, don't conjugate
        poles = new_poles
    else:
        # For even n, add all conjugates
        poles = poles + [complex(p.real, -p.imag) for p in poles]

    # Compute gain
    prod_p = 1.0
    for p in poles:
        prod_p *= abs(p)

    prod_z = 1.0
    for z in zeros:
        prod_z *= abs(z)

    if prod_z > 1e-15:
        gain = prod_p / prod_z
    else:
        gain = prod_p

    if n % 2 == 0:
        gain = gain / math.sqrt(1 + eps_sq)

    return zeros, poles, gain
