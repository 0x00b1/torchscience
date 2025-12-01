import torch
from torch import Tensor


def incomplete_legendre_elliptic_integral_pi(
    n: Tensor, phi: Tensor, m: Tensor
) -> Tensor:
    r"""
    Incomplete elliptic integral of the third kind.

    Computes the Legendre incomplete elliptic integral of the third kind
    Pi(n, phi, m).

    Mathematical Definition
    -----------------------
    The incomplete elliptic integral of the third kind is defined as:

    .. math::

       \Pi(n, \phi, m) = \int_0^{\phi}
       \frac{d\theta}{(1 - n \sin^2 \theta) \sqrt{1 - m \sin^2 \theta}}

    where n is the characteristic, phi is the amplitude, and m is the parameter
    (m = k^2 where k is the elliptic modulus).

    Relation to Carlson Integrals
    -----------------------------
    The integral is computed using Carlson symmetric forms:

    .. math::

       \Pi(n, \phi, m) = \sin(\phi) \, R_F(\cos^2 \phi, 1-m\sin^2 \phi, 1)
                         + \frac{n}{3} \sin^3(\phi) \, R_J(\cos^2 \phi,
                           1-m\sin^2 \phi, 1, 1-n\sin^2 \phi)

    Domain
    ------
    - n: real number (characteristic parameter)
    - phi: real number (amplitude, in radians)
    - m: real number, typically 0 <= m <= 1 (parameter)
    - For real computation: 1 - n*sin^2(phi) > 0 to avoid singularity

    Special Values
    --------------
    - Pi(n, 0, m) = 0
    - Pi(0, phi, m) = F(phi, m) (incomplete elliptic integral of the first kind)
    - Pi(n, pi/2, m) = Pi(n, m) (complete elliptic integral of the third kind)
    - Pi(0, phi, 0) = phi
    - Pi(n, phi, 0) = atan(sqrt(1-n) * tan(phi)) / sqrt(1-n) for n < 1

    Limiting Cases
    --------------
    - As phi -> 0: Pi(n, phi, m) -> 0
    - As n -> 0: Pi(n, phi, m) -> F(phi, m)
    - As m -> 0: Pi(n, phi, m) -> atan(sqrt(1-n) * tan(phi)) / sqrt(1-n)

    Algorithm
    ---------
    Uses Carlson's algorithms for R_F and R_J which employ the duplication
    theorem. The iteration converges quadratically.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Incomplete elliptic integrals of the third kind appear in:
    - Pendulum motion with finite amplitude
    - Geodesics on ellipsoids
    - Arc length of ellipses
    - Magnetic field calculations for current loops
    - Conformal mapping problems
    - Potential theory
    - Spacecraft trajectory optimization
    - Seiffert spiral calculations

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    Gradients are computed using numerical finite differences.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    n : Tensor
        Characteristic parameter. Broadcasting with phi and m is supported.
    phi : Tensor
        Amplitude in radians. Broadcasting with n and m is supported.
    m : Tensor
        Parameter (m = k^2 where k is the modulus).
        Broadcasting with n and phi is supported.

    Returns
    -------
    Tensor
        Incomplete elliptic integral of the third kind Pi(n, phi, m) evaluated
        at the input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> n = torch.tensor([0.5])
    >>> phi = torch.tensor([1.0])
    >>> m = torch.tensor([0.3])
    >>> incomplete_legendre_elliptic_integral_pi(n, phi, m)
    tensor([1.2789])

    When phi = pi/2, reduces to complete Pi:

    >>> import math
    >>> n = torch.tensor([0.5])
    >>> phi = torch.tensor([math.pi / 2])
    >>> m = torch.tensor([0.3])
    >>> incomplete_val = incomplete_legendre_elliptic_integral_pi(n, phi, m)
    >>> # Compare with complete Pi(n, m)
    >>> import torchscience
    >>> complete_val = torchscience.special_functions.complete_legendre_elliptic_integral_pi(n, m)
    >>> torch.allclose(incomplete_val, complete_val, rtol=1e-5)
    True

    When n = 0, reduces to F(phi, m):

    >>> n = torch.tensor([0.0])
    >>> phi = torch.tensor([1.0])
    >>> m = torch.tensor([0.5])
    >>> pi_val = incomplete_legendre_elliptic_integral_pi(n, phi, m)
    >>> # Compare with F(phi, m) = sin(phi) * R_F(cos^2(phi), 1-m*sin^2(phi), 1)
    >>> import torchscience
    >>> sin_phi = torch.sin(phi)
    >>> cos_phi = torch.cos(phi)
    >>> f_val = sin_phi * torchscience.special_functions.carlson_elliptic_integral_r_f(
    ...     cos_phi**2, 1 - m * sin_phi**2, torch.tensor([1.0]))
    >>> torch.allclose(pi_val, f_val, rtol=1e-5)
    True

    Autograd:

    >>> n = torch.tensor([0.3], requires_grad=True)
    >>> phi = torch.tensor([1.0])
    >>> m = torch.tensor([0.5])
    >>> result = incomplete_legendre_elliptic_integral_pi(n, phi, m)
    >>> result.backward()
    >>> n.grad  # Gradient w.r.t. n
    tensor([0.4962])

    .. warning:: Singularities

       When 1 - n*sin^2(phi) = 0 or 1 - m*sin^2(phi) = 0, the integral has
       singularities and the function may return inf or produce numerical
       instability.

    Notes
    -----
    - The parameter convention uses m = k^2 (not the modulus k directly).
    - Some references use different conventions for n (opposite sign).
    - The amplitude phi is in radians.
    - For the complete elliptic integral (phi = pi/2), use
      complete_legendre_elliptic_integral_pi instead.

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_j : Carlson's elliptic integral R_J
    complete_legendre_elliptic_integral_pi : Complete elliptic integral Pi
    complete_legendre_elliptic_integral_e : Complete elliptic integral E
    complete_legendre_elliptic_integral_k : Complete elliptic integral K
    """
    return torch.ops.torchscience.incomplete_legendre_elliptic_integral_pi(
        n, phi, m
    )
