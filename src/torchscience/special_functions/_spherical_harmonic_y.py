"""Spherical harmonics Y_l^m(theta, phi)."""

import math

import torch
from torch import Tensor

from ._associated_legendre_polynomial_p import associated_legendre_polynomial_p


def spherical_harmonic_y(
    l: int,
    m: int,
    theta: Tensor,
    phi: Tensor,
    real: bool = False,
) -> Tensor:
    r"""Compute spherical harmonic Y_l^m(theta, phi).

    Mathematical Definition
    -----------------------
    The complex spherical harmonic is defined as:

    .. math::

        Y_l^m(\theta, \phi) = N_l^m P_l^m(\cos\theta) e^{im\phi}

    where the normalization factor is:

    .. math::

        N_l^m = \sqrt{\frac{(2l+1)}{4\pi} \frac{(l-m)!}{(l+m)!}}

    For real spherical harmonics:

    .. math::

        Y_l^m = \begin{cases}
            \sqrt{2} N_l^{|m|} P_l^{|m|}(\cos\theta) \cos(|m|\phi) & m > 0 \\
            N_l^0 P_l^0(\cos\theta) & m = 0 \\
            \sqrt{2} N_l^{|m|} P_l^{|m|}(\cos\theta) \sin(|m|\phi) & m < 0
        \end{cases}

    Parameters
    ----------
    l : int
        Degree of the spherical harmonic. Must be non-negative.
    m : int
        Order of the spherical harmonic. Must satisfy |m| <= l.
    theta : Tensor
        Polar angle (colatitude) in radians, range [0, pi].
    phi : Tensor
        Azimuthal angle in radians, range [0, 2*pi].
    real : bool, optional
        If True, return real spherical harmonics. Default is False (complex).

    Returns
    -------
    Tensor
        Spherical harmonic values at the given angles.
        Complex tensor if real=False, real tensor if real=True.

    Raises
    ------
    ValueError
        If l < 0 or |m| > l.

    Examples
    --------
    >>> theta = torch.tensor([0.0, math.pi/4, math.pi/2])
    >>> phi = torch.tensor([0.0, math.pi/4, math.pi/2])
    >>> Y_00 = spherical_harmonic_y(0, 0, theta, phi)
    >>> # Y_0^0 = 1/(2*sqrt(pi)) for all angles

    >>> Y_10 = spherical_harmonic_y(1, 0, theta, phi)
    >>> # Y_1^0 = sqrt(3/(4*pi)) * cos(theta)

    Notes
    -----
    Uses the Condon-Shortley phase convention, which includes the (-1)^m
    factor in the associated Legendre polynomials.

    See Also
    --------
    spherical_harmonic_y_all : Compute all Y_l^m for l=0..l_max
    associated_legendre_polynomial_p : Associated Legendre polynomials
    """
    if l < 0:
        raise ValueError(f"l must be non-negative, got {l}")

    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m}, l={l}")

    # Compute cos(theta) for the Legendre polynomial
    cos_theta = torch.cos(theta)

    # Get associated Legendre polynomial P_l^|m|(cos(theta))
    abs_m = abs(m)
    P_lm = associated_legendre_polynomial_p(l, abs_m, cos_theta)

    # Normalization factor: sqrt((2l+1)/(4*pi) * (l-|m|)!/(l+|m|)!)
    norm = math.sqrt(
        (2 * l + 1)
        / (4 * math.pi)
        * math.factorial(l - abs_m)
        / math.factorial(l + abs_m)
    )

    if real:
        # Real spherical harmonics
        if m > 0:
            # Y_l^m = sqrt(2) * N * P_l^m * cos(m*phi)
            return math.sqrt(2) * norm * P_lm * torch.cos(m * phi)
        elif m == 0:
            # Y_l^0 = N * P_l^0
            return norm * P_lm
        else:
            # Y_l^{-m} = sqrt(2) * N * P_l^|m| * sin(|m|*phi)
            return math.sqrt(2) * norm * P_lm * torch.sin(abs_m * phi)
    else:
        # Complex spherical harmonics
        # Y_l^m = N * P_l^|m| * exp(i*m*phi)

        # Handle negative m using symmetry:
        # Y_l^{-m} = (-1)^m * conj(Y_l^m)
        # But we can compute directly using P_l^|m| and exp(i*m*phi)

        # For m >= 0: Y_l^m = N * P_l^m * exp(i*m*phi)
        # For m < 0: Y_l^m = (-1)^|m| * N * P_l^|m| * exp(i*m*phi)
        #          = (-1)^|m| * N * P_l^|m| * exp(-i*|m|*phi)

        # Since we're using Condon-Shortley convention, P_l^m already has (-1)^m
        # For negative m, we need to adjust

        if m >= 0:
            phase = torch.exp(1j * m * phi)
            return norm * P_lm.to(phase.dtype) * phase
        else:
            # Y_l^{-m} = (-1)^m * conj(Y_l^m)
            sign = (-1) ** abs_m
            phase = torch.exp(-1j * abs_m * phi)
            return sign * norm * P_lm.to(phase.dtype) * phase


def spherical_harmonic_y_cartesian(
    l: int,
    m: int,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    real: bool = False,
) -> Tensor:
    r"""Compute spherical harmonic Y_l^m from Cartesian coordinates.

    Assumes the input points lie on the unit sphere (x^2 + y^2 + z^2 = 1).

    Parameters
    ----------
    l : int
        Degree of the spherical harmonic. Must be non-negative.
    m : int
        Order of the spherical harmonic. Must satisfy |m| <= l.
    x : Tensor
        X coordinate on unit sphere.
    y : Tensor
        Y coordinate on unit sphere.
    z : Tensor
        Z coordinate on unit sphere.
    real : bool, optional
        If True, return real spherical harmonics. Default is False (complex).

    Returns
    -------
    Tensor
        Spherical harmonic values at the given points.

    Notes
    -----
    The conversion from Cartesian to spherical coordinates is:
    - theta = arccos(z) (polar angle)
    - phi = atan2(y, x) (azimuthal angle)

    Examples
    --------
    >>> # Point on the unit sphere at north pole
    >>> x, y, z = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0])
    >>> Y = spherical_harmonic_y_cartesian(0, 0, x, y, z)
    """
    # Convert to spherical coordinates
    # theta = arccos(z) for unit sphere
    # phi = atan2(y, x)
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x)

    return spherical_harmonic_y(l, m, theta, phi, real=real)


def spherical_harmonic_y_all(
    l_max: int,
    theta: Tensor,
    phi: Tensor,
    real: bool = False,
) -> Tensor:
    r"""Compute all spherical harmonics Y_l^m for l=0..l_max, m=-l..l.

    Parameters
    ----------
    l_max : int
        Maximum degree. Returns harmonics for l=0, 1, ..., l_max.
    theta : Tensor
        Polar angle (colatitude) in radians, shape (...).
    phi : Tensor
        Azimuthal angle in radians, shape (...).
    real : bool, optional
        If True, return real spherical harmonics. Default is False (complex).

    Returns
    -------
    Tensor
        Tensor of shape (..., (l_max+1)^2) containing all spherical harmonics.
        Harmonics are ordered as: Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ...

    Notes
    -----
    The indexing follows the convention:
    - index = l^2 + l + m

    For l_max=2, the ordering is:
    - index 0: Y_0^0
    - index 1: Y_1^{-1}
    - index 2: Y_1^0
    - index 3: Y_1^1
    - index 4: Y_2^{-2}
    - index 5: Y_2^{-1}
    - index 6: Y_2^0
    - index 7: Y_2^1
    - index 8: Y_2^2

    Examples
    --------
    >>> theta = torch.tensor([0.0, math.pi/2])
    >>> phi = torch.tensor([0.0, math.pi/4])
    >>> Y_all = spherical_harmonic_y_all(2, theta, phi)
    >>> Y_all.shape
    torch.Size([2, 9])
    """
    if l_max < 0:
        raise ValueError(f"l_max must be non-negative, got {l_max}")

    batch_shape = theta.shape
    n_harmonics = (l_max + 1) ** 2

    # Determine output dtype
    if real:
        dtype = torch.promote_types(theta.dtype, phi.dtype)
    else:
        # Complex output
        base_dtype = torch.promote_types(theta.dtype, phi.dtype)
        if base_dtype == torch.float32:
            dtype = torch.complex64
        else:
            dtype = torch.complex128

    result = torch.zeros(
        batch_shape + (n_harmonics,), dtype=dtype, device=theta.device
    )

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l * l + l + m
            Y_lm = spherical_harmonic_y(l, m, theta, phi, real=real)
            result[..., idx] = Y_lm

    return result
