import torch
from torch import Tensor


def complete_legendre_elliptic_integral_pi(n: Tensor, m: Tensor) -> Tensor:
    r"""
    Complete elliptic integral of the third kind.

    Computes the Legendre complete elliptic integral of the third kind Pi(n, m).

    Mathematical Definition
    -----------------------
    The complete elliptic integral of the third kind is defined as:

    .. math::

       \Pi(n, m) = \int_0^{\pi/2}
       \frac{d\theta}{(1 - n \sin^2 \theta) \sqrt{1 - m \sin^2 \theta}}

    where n is the characteristic and m is the parameter (m = k^2 where k is
    the elliptic modulus).

    Relation to Carlson Integrals
    -----------------------------
    The integral is computed using Carlson symmetric forms:

    .. math::

       \Pi(n, m) = R_F(0, 1-m, 1) + \frac{n}{3} R_J(0, 1-m, 1, 1-n)

    Domain
    ------
    - n: real number (characteristic parameter)
    - m: real number, typically 0 <= m <= 1 (parameter)
    - For real computation: n < 1 or special handling needed at n = 1

    Special Values
    --------------
    - Pi(0, m) = K(m) (complete elliptic integral of the first kind)
    - Pi(n, 0) = pi / (2 * sqrt(1 - n)) for n < 1
    - Pi(m, m) = E(m) / (1 - m) for m < 1
    - Pi(n, 1) has a logarithmic singularity

    Limiting Cases
    --------------
    - As n -> 0: Pi(n, m) -> K(m)
    - As m -> 0: Pi(n, m) -> pi / (2 * sqrt(1 - n))
    - As n -> 1 from below: Pi(n, m) -> infinity

    Algorithm
    ---------
    Uses Carlson's algorithms for R_F and R_J which employ the duplication
    theorem. The iteration converges quadratically.

    - Provides consistent results across CPU and CUDA devices.
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy.

    Applications
    ------------
    Complete elliptic integrals of the third kind appear in:
    - Pendulum motion with finite amplitude
    - Geodesics on ellipsoids
    - Magnetic field calculations for current loops
    - Conformal mapping problems
    - Potential theory
    - Spacecraft trajectory optimization

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
        Characteristic parameter. Broadcasting with m is supported.
    m : Tensor
        Parameter (m = k^2 where k is the modulus).
        Broadcasting with n is supported.

    Returns
    -------
    Tensor
        Complete elliptic integral of the third kind Pi(n, m) evaluated at
        the input values. Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> n = torch.tensor([0.5])
    >>> m = torch.tensor([0.3])
    >>> complete_legendre_elliptic_integral_pi(n, m)
    tensor([2.3254])

    When n = 0, reduces to K(m):

    >>> n = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> pi_val = complete_legendre_elliptic_integral_pi(n, m)
    >>> # Compare with K(m) = R_F(0, 1-m, 1)
    >>> import torchscience
    >>> k_val = torchscience.special_functions.carlson_elliptic_integral_r_f(
    ...     torch.tensor([0.0]), torch.tensor([0.5]), torch.tensor([1.0]))
    >>> torch.allclose(pi_val, k_val)
    True

    Autograd:

    >>> n = torch.tensor([0.3], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> result = complete_legendre_elliptic_integral_pi(n, m)
    >>> result.backward()
    >>> n.grad  # Gradient w.r.t. n
    tensor([1.3542])

    .. warning:: Singularities

       When n = 1 or m = 1, the integral has singularities and the function
       may return inf or produce numerical instability.

    Notes
    -----
    - The parameter convention uses m = k^2 (not the modulus k directly).
    - Some references use different conventions for n (opposite sign).
    - For incomplete elliptic integrals of the third kind, a separate
      function would be needed.

    See Also
    --------
    carlson_elliptic_integral_r_f : Carlson's elliptic integral R_F
    carlson_elliptic_integral_r_j : Carlson's elliptic integral R_J
    complete_legendre_elliptic_integral_e : Complete elliptic integral E
    """
    return torch.ops.torchscience.complete_legendre_elliptic_integral_pi(n, m)
