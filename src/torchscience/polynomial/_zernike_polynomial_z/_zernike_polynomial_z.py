"""Zernike polynomials Z_n^m(rho, theta).

Zernike polynomials are orthogonal polynomials defined on the unit disk,
widely used in optics for describing wavefront aberrations.
"""

import math

import torch
from torch import Tensor


def zernike_polynomial_z_radial(n: int, m: int, rho: Tensor) -> Tensor:
    r"""Compute the radial Zernike polynomial R_n^m(rho).

    Mathematical Definition
    -----------------------
    The radial Zernike polynomial is defined as:

    .. math::

        R_n^m(\rho) = \sum_{s=0}^{(n-m)/2} \frac{(-1)^s (n-s)!}
        {s! \left(\frac{n+m}{2}-s\right)! \left(\frac{n-m}{2}-s\right)!}
        \rho^{n-2s}

    This is valid when n - |m| is even and non-negative.

    Parameters
    ----------
    n : int
        Radial degree. Must be non-negative.
    m : int
        Azimuthal degree. Must satisfy |m| <= n and n - |m| even.
    rho : Tensor
        Radial coordinate, typically in [0, 1].

    Returns
    -------
    Tensor
        Values of R_n^m(rho).

    Raises
    ------
    ValueError
        If n < 0, |m| > n, or n - |m| is odd.

    Examples
    --------
    >>> rho = torch.tensor([0.0, 0.5, 1.0])
    >>> R_00 = zernike_polynomial_z_radial(0, 0, rho)  # R_0^0 = 1
    >>> R_20 = zernike_polynomial_z_radial(2, 0, rho)  # R_2^0 = 2*rho^2 - 1
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    abs_m = abs(m)
    if abs_m > n:
        raise ValueError(f"|m| must be <= n, got m={m}, n={n}")

    if (n - abs_m) % 2 != 0:
        raise ValueError(f"n - |m| must be even, got n={n}, m={m}")

    # Compute R_n^|m|(rho) using the explicit sum formula
    k_max = (n - abs_m) // 2
    result = torch.zeros_like(rho)

    for s in range(k_max + 1):
        # Coefficient: (-1)^s * (n-s)! / (s! * ((n+|m|)/2-s)! * ((n-|m|)/2-s)!)
        sign = (-1) ** s
        numerator = math.factorial(n - s)
        denominator = (
            math.factorial(s)
            * math.factorial((n + abs_m) // 2 - s)
            * math.factorial((n - abs_m) // 2 - s)
        )
        coeff = sign * numerator / denominator
        result = result + coeff * rho ** (n - 2 * s)

    return result


def zernike_polynomial_z(
    n: int, m: int, rho: Tensor, theta: Tensor, normalized: bool = True
) -> Tensor:
    r"""Compute Zernike polynomial Z_n^m(rho, theta).

    Mathematical Definition
    -----------------------
    The Zernike polynomial is defined as:

    .. math::

        Z_n^m(\rho, \theta) = \begin{cases}
            N_n^m R_n^{|m|}(\rho) \cos(m\theta) & m \geq 0 \\
            -N_n^m R_n^{|m|}(\rho) \sin(m\theta) & m < 0
        \end{cases}

    where N_n^m is a normalization factor:

    .. math::

        N_n^m = \sqrt{\frac{2(n+1)}{1 + \delta_{m0}}}

    where delta_{m0} is the Kronecker delta.

    Parameters
    ----------
    n : int
        Radial degree. Must be non-negative.
    m : int
        Azimuthal degree. Must satisfy |m| <= n and n - |m| even.
    rho : Tensor
        Radial coordinate, typically in [0, 1].
    theta : Tensor
        Azimuthal angle in radians.
    normalized : bool, optional
        If True (default), include normalization factor N_n^m.

    Returns
    -------
    Tensor
        Values of Z_n^m(rho, theta).

    Examples
    --------
    >>> rho = torch.tensor([0.5, 0.8, 1.0])
    >>> theta = torch.tensor([0.0, math.pi/4, math.pi/2])
    >>> Z_00 = zernike_polynomial_z(0, 0, rho, theta)  # Piston
    >>> Z_11 = zernike_polynomial_z(1, 1, rho, theta)  # Tilt X

    Notes
    -----
    The convention used here follows the standard in optics, where:
    - Positive m corresponds to cosine (symmetric about y-axis)
    - Negative m corresponds to sine (antisymmetric about y-axis)

    See Also
    --------
    zernike_polynomial_z_radial : Radial part only
    zernike_polynomial_z_noll : Using Noll indexing
    zernike_polynomial_z_osa : Using OSA/ANSI indexing
    """
    # Get radial polynomial
    R_nm = zernike_polynomial_z_radial(n, m, rho)

    # Angular part
    if m >= 0:
        angular = torch.cos(m * theta)
    else:
        angular = -torch.sin(abs(m) * theta)

    # Normalization factor
    if normalized:
        if m == 0:
            norm = math.sqrt(n + 1)
        else:
            norm = math.sqrt(2 * (n + 1))
    else:
        norm = 1.0

    return norm * R_nm * angular


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index to (n, m) indices.

    The Noll indexing convention starts at j=1 and orders Zernike
    polynomials in a specific sequence commonly used in optics.

    Parameters
    ----------
    j : int
        Noll index, starting from 1.

    Returns
    -------
    tuple[int, int]
        The (n, m) indices.

    Raises
    ------
    ValueError
        If j < 1.

    Examples
    --------
    >>> noll_to_nm(1)  # Piston
    (0, 0)
    >>> noll_to_nm(2)  # Tilt Y
    (1, 1)
    >>> noll_to_nm(3)  # Tilt X
    (1, -1)
    >>> noll_to_nm(4)  # Defocus
    (2, 0)
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")

    # Find n: the radial degree
    # n satisfies n(n+1)/2 < j <= (n+1)(n+2)/2
    n = int((-3 + math.sqrt(9 + 8 * j)) / 2)
    if (n + 1) * (n + 2) // 2 < j:
        n += 1

    # Position within the row (1-indexed)
    k = j - n * (n + 1) // 2

    # Compute |m| based on position within the row
    # For row n, the sequence of |m| values is:
    # - Even n: 0, 2, 2, 4, 4, ... (m=0 appears once, others twice)
    # - Odd n: 1, 1, 3, 3, 5, 5, ... (all appear twice)
    if n % 2 == 0:
        # Even n: |m| = 0 for k=1, then |m| = 2*ceil((k-1)/2) for k>1
        if k == 1:
            abs_m = 0
        else:
            abs_m = 2 * ((k + 1) // 2 - 1) + 2
            if abs_m > n:
                abs_m = n
    else:
        # Odd n: |m| = 2*ceil(k/2) - 1
        abs_m = 2 * ((k + 1) // 2) - 1
        if abs_m > n:
            abs_m = n

    # Determine sign of m based on j parity
    # Even j -> positive m, odd j -> negative m
    if abs_m == 0:
        m = 0
    elif j % 2 == 0:
        m = abs_m
    else:
        m = -abs_m

    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """Convert (n, m) indices to Noll index.

    Parameters
    ----------
    n : int
        Radial degree.
    m : int
        Azimuthal degree.

    Returns
    -------
    int
        Noll index j.

    Raises
    ------
    ValueError
        If the (n, m) pair is invalid.

    Examples
    --------
    >>> nm_to_noll(0, 0)  # Piston
    1
    >>> nm_to_noll(1, 1)  # Tilt Y
    2
    >>> nm_to_noll(1, -1)  # Tilt X
    3
    >>> nm_to_noll(2, 0)  # Defocus
    4
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if abs(m) > n:
        raise ValueError(f"|m| must be <= n, got n={n}, m={m}")
    if (n - abs(m)) % 2 != 0:
        raise ValueError(f"n - |m| must be even, got n={n}, m={m}")

    # Base index for row n
    base = n * (n + 1) // 2

    # In Noll convention: even j -> positive m, odd j -> negative m
    # We need to find k such that j = base + k has the right parity

    if m == 0:
        # m=0 appears only once per row (for even n only)
        k = 1
    else:
        abs_m = abs(m)
        # Find the two possible positions for this |m|
        if n % 2 == 0:
            # Even n: |m|=2p appears at positions 2p and 2p+1
            p = abs_m // 2
            first_k = 2 * p
            second_k = 2 * p + 1
        else:
            # Odd n: |m|=2p-1 appears at positions 2p-1 and 2p
            p = (abs_m + 1) // 2
            first_k = 2 * p - 1
            second_k = 2 * p

        # Choose k based on desired parity of j = base + k
        # m > 0 -> want j even -> want (base + k) even -> k has same parity as base
        # m < 0 -> want j odd -> want (base + k) odd -> k has opposite parity to base
        base_odd = base % 2 == 1

        if m > 0:
            # Want (base + k) even
            if base_odd:
                # Need k odd
                k = first_k if first_k % 2 == 1 else second_k
            else:
                # Need k even
                k = first_k if first_k % 2 == 0 else second_k
        else:
            # Want (base + k) odd
            if base_odd:
                # Need k even
                k = first_k if first_k % 2 == 0 else second_k
            else:
                # Need k odd
                k = first_k if first_k % 2 == 1 else second_k

    return base + k


def osa_to_nm(j: int) -> tuple[int, int]:
    """Convert OSA/ANSI index to (n, m) indices.

    The OSA (Optical Society of America) / ANSI indexing convention
    starts at j=0 and uses a different ordering than Noll.

    Parameters
    ----------
    j : int
        OSA/ANSI index, starting from 0.

    Returns
    -------
    tuple[int, int]
        The (n, m) indices.

    Raises
    ------
    ValueError
        If j < 0.

    Examples
    --------
    >>> osa_to_nm(0)  # Piston
    (0, 0)
    >>> osa_to_nm(1)  # Tilt Y
    (1, -1)
    >>> osa_to_nm(2)  # Tilt X
    (1, 1)
    >>> osa_to_nm(3)  # Defocus
    (2, -2)
    """
    if j < 0:
        raise ValueError(f"OSA index must be >= 0, got {j}")

    # Find n such that n*(n+1)/2 <= j < (n+1)*(n+2)/2
    n = 0
    while (n + 1) * (n + 2) // 2 <= j:
        n += 1

    # Position within row
    pos = j - n * (n + 1) // 2

    # m values for row n: -n, -n+2, ..., n-2, n
    m = -n + 2 * pos

    return n, m


def nm_to_osa(n: int, m: int) -> int:
    """Convert (n, m) indices to OSA/ANSI index.

    Parameters
    ----------
    n : int
        Radial degree.
    m : int
        Azimuthal degree.

    Returns
    -------
    int
        OSA/ANSI index j.

    Raises
    ------
    ValueError
        If the (n, m) pair is invalid.

    Examples
    --------
    >>> nm_to_osa(0, 0)  # Piston
    0
    >>> nm_to_osa(1, -1)  # Tilt Y
    1
    >>> nm_to_osa(1, 1)  # Tilt X
    2
    >>> nm_to_osa(2, -2)
    3
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if abs(m) > n:
        raise ValueError(f"|m| must be <= n, got n={n}, m={m}")
    if (n - abs(m)) % 2 != 0:
        raise ValueError(f"n - |m| must be even, got n={n}, m={m}")

    # Base index for row n
    base = n * (n + 1) // 2

    # Position within row: m = -n + 2*pos => pos = (m + n) / 2
    pos = (m + n) // 2

    return base + pos


def zernike_polynomial_z_noll(
    j: int, rho: Tensor, theta: Tensor, normalized: bool = True
) -> Tensor:
    """Compute Zernike polynomial using Noll index.

    Parameters
    ----------
    j : int
        Noll index, starting from 1.
    rho : Tensor
        Radial coordinate.
    theta : Tensor
        Azimuthal angle in radians.
    normalized : bool, optional
        If True (default), include normalization factor.

    Returns
    -------
    Tensor
        Values of Z_j(rho, theta).

    Examples
    --------
    >>> rho = torch.tensor([0.5, 0.8])
    >>> theta = torch.tensor([0.0, math.pi/4])
    >>> Z_1 = zernike_polynomial_z_noll(1, rho, theta)  # Piston
    >>> Z_4 = zernike_polynomial_z_noll(4, rho, theta)  # Defocus
    """
    n, m = noll_to_nm(j)
    return zernike_polynomial_z(n, m, rho, theta, normalized=normalized)


def zernike_polynomial_z_osa(
    j: int, rho: Tensor, theta: Tensor, normalized: bool = True
) -> Tensor:
    """Compute Zernike polynomial using OSA/ANSI index.

    Parameters
    ----------
    j : int
        OSA/ANSI index, starting from 0.
    rho : Tensor
        Radial coordinate.
    theta : Tensor
        Azimuthal angle in radians.
    normalized : bool, optional
        If True (default), include normalization factor.

    Returns
    -------
    Tensor
        Values of Z_j(rho, theta).

    Examples
    --------
    >>> rho = torch.tensor([0.5, 0.8])
    >>> theta = torch.tensor([0.0, math.pi/4])
    >>> Z_0 = zernike_polynomial_z_osa(0, rho, theta)  # Piston
    >>> Z_3 = zernike_polynomial_z_osa(3, rho, theta)  # Oblique astigmatism
    """
    n, m = osa_to_nm(j)
    return zernike_polynomial_z(n, m, rho, theta, normalized=normalized)


def zernike_polynomial_z_all(
    n_max: int, rho: Tensor, theta: Tensor, normalized: bool = True
) -> Tensor:
    """Compute all Zernike polynomials up to radial degree n_max.

    Parameters
    ----------
    n_max : int
        Maximum radial degree.
    rho : Tensor
        Radial coordinate, shape (...).
    theta : Tensor
        Azimuthal angle in radians, shape (...).
    normalized : bool, optional
        If True (default), include normalization factors.

    Returns
    -------
    Tensor
        Tensor of shape (..., (n_max+1)*(n_max+2)/2) containing all Zernike
        polynomials ordered by OSA/ANSI convention.

    Examples
    --------
    >>> rho = torch.tensor([0.5, 0.8])
    >>> theta = torch.tensor([0.0, math.pi/4])
    >>> Z_all = zernike_polynomial_z_all(2, rho, theta)
    >>> Z_all.shape
    torch.Size([2, 6])  # 6 polynomials for n_max=2
    """
    if n_max < 0:
        raise ValueError(f"n_max must be non-negative, got {n_max}")

    n_polys = (n_max + 1) * (n_max + 2) // 2
    batch_shape = rho.shape
    dtype = torch.promote_types(rho.dtype, theta.dtype)

    result = torch.zeros(
        batch_shape + (n_polys,), dtype=dtype, device=rho.device
    )

    idx = 0
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            Z_nm = zernike_polynomial_z(
                n, m, rho, theta, normalized=normalized
            )
            result[..., idx] = Z_nm
            idx += 1

    return result


def zernike_polynomial_z_fit(
    data: Tensor,
    rho: Tensor,
    theta: Tensor,
    n_max: int,
    normalized: bool = True,
) -> Tensor:
    """Fit Zernike coefficients to wavefront data.

    Given wavefront data at points (rho, theta), compute the Zernike
    coefficients that best fit the data in a least-squares sense.

    Parameters
    ----------
    data : Tensor
        Wavefront data values, shape (..., n_points).
    rho : Tensor
        Radial coordinates of data points, shape (n_points,).
    theta : Tensor
        Azimuthal angles of data points, shape (n_points,).
    n_max : int
        Maximum radial degree for fitting.
    normalized : bool, optional
        If True (default), use normalized Zernike polynomials.

    Returns
    -------
    Tensor
        Zernike coefficients, shape (..., (n_max+1)*(n_max+2)/2).

    Notes
    -----
    The fit uses least-squares regression. For better numerical stability
    with many terms, consider using a regularized approach.

    Examples
    --------
    >>> # Generate synthetic wavefront with known Zernike coefficients
    >>> rho = torch.rand(100)
    >>> theta = torch.rand(100) * 2 * math.pi
    >>> true_coeffs = torch.tensor([1.0, 0.5, -0.3, 0.2, 0.0, 0.1])
    >>> data = zernike_polynomial_z_all(2, rho, theta) @ true_coeffs
    >>> fitted_coeffs = zernike_polynomial_z_fit(data, rho, theta, n_max=2)
    """
    # Build design matrix
    Z_matrix = zernike_polynomial_z_all(
        n_max, rho, theta, normalized=normalized
    )

    # Solve least squares: Z_matrix @ coeffs = data
    # Using normal equations: coeffs = (Z^T Z)^{-1} Z^T data
    # But lstsq is more numerically stable
    if data.dim() == 1:
        coeffs = torch.linalg.lstsq(
            Z_matrix, data.unsqueeze(-1)
        ).solution.squeeze(-1)
    else:
        # Batch case
        batch_shape = data.shape[:-1]
        n_points = data.shape[-1]
        data_flat = data.reshape(-1, n_points)

        # Z_matrix is (n_points, n_polys), data is (batch, n_points)
        # We need to solve for each batch element
        coeffs_list = []
        for i in range(data_flat.shape[0]):
            c = torch.linalg.lstsq(
                Z_matrix, data_flat[i].unsqueeze(-1)
            ).solution.squeeze(-1)
            coeffs_list.append(c)
        coeffs = torch.stack(coeffs_list).reshape(
            batch_shape + (Z_matrix.shape[-1],)
        )

    return coeffs


# Common Zernike polynomial names (Noll convention)
ZERNIKE_NAMES = {
    1: "Piston",
    2: "Tilt Y",
    3: "Tilt X",
    4: "Defocus",
    5: "Astigmatism 45°",
    6: "Astigmatism 0°",
    7: "Coma Y",
    8: "Coma X",
    9: "Trefoil Y",
    10: "Trefoil X",
    11: "Spherical",
    12: "Secondary Astigmatism 45°",
    13: "Secondary Astigmatism 0°",
    14: "Secondary Coma Y",
    15: "Secondary Coma X",
    16: "Tetrafoil Y",
    17: "Tetrafoil X",
    18: "Secondary Trefoil Y",
    19: "Secondary Trefoil X",
    20: "Pentafoil Y",
    21: "Pentafoil X",
    22: "Secondary Spherical",
}
