import torch
from torch import Tensor


def zernike_polynomial_r(n: Tensor, m: Tensor, rho: Tensor) -> Tensor:
    r"""
    Radial Zernike polynomial.

    Computes the radial Zernike polynomial :math:`R_n^m(\rho)`.

    Mathematical Definition
    -----------------------
    The radial Zernike polynomial is defined for :math:`n \geq m \geq 0` with
    :math:`(n-m)` even:

    .. math::

       R_n^m(\rho) = (-1)^{(n-m)/2} \rho^m P_k^{(m,0)}(1 - 2\rho^2)

    where :math:`k = (n-m)/2` and :math:`P_k^{(\alpha,\beta)}(x)` is the Jacobi
    polynomial.

    Alternatively, via the explicit sum:

    .. math::

       R_n^m(\rho) = \sum_{s=0}^{(n-m)/2} (-1)^s
                     \frac{(n-s)!}{s! \left(\frac{n+m}{2}-s\right)!
                     \left(\frac{n-m}{2}-s\right)!} \rho^{n-2s}

    Special Values
    --------------
    - :math:`R_n^m(0) = 0` for :math:`m > 0`
    - :math:`R_n^0(0) = (-1)^{n/2}` for :math:`n` even
    - :math:`R_n^m(1) = 1` for all valid :math:`n, m`
    - :math:`R_n^n(\rho) = \rho^n`

    Constraints
    -----------
    - :math:`n \geq m \geq 0`
    - :math:`(n - m)` must be even

    For invalid combinations (:math:`n < m` or :math:`(n-m)` odd), returns 0.

    Applications
    ------------
    - **Optics**: Wavefront analysis and aberration characterization in
      optical systems. Zernike polynomials form an orthogonal basis on the
      unit disk, making them ideal for representing wavefront aberrations.

    - **Image processing**: Zernike moments are used for rotation-invariant
      pattern recognition and image analysis.

    - **Optical testing**: Standard representation for optical aberrations
      including defocus, astigmatism, coma, and spherical aberration.

    Parameters
    ----------
    n : Tensor
        Radial order (degree) of the polynomial. Must be a non-negative
        integer. Broadcasting with m and rho is supported.
    m : Tensor
        Azimuthal frequency (angular order). Must satisfy :math:`|m| \leq n`
        and :math:`(n - |m|)` must be even. The absolute value is used
        internally. Broadcasting with n and rho is supported.
    rho : Tensor
        Radial coordinate, typically in the range [0, 1] for the unit disk.
        Broadcasting with n and m is supported.

    Returns
    -------
    Tensor
        The radial Zernike polynomial :math:`R_n^m(\rho)` evaluated at the
        input values.

    Examples
    --------
    Basic Zernike polynomials:

    >>> n = torch.tensor([0.0, 1.0, 2.0, 2.0])
    >>> m = torch.tensor([0.0, 1.0, 0.0, 2.0])
    >>> rho = torch.tensor([0.5])
    >>> zernike_polynomial_r(n, m, rho)
    tensor([ 1.0000,  0.5000, -0.5000,  0.2500])

    Verify R_2^0(rho) = 2*rho^2 - 1:

    >>> n = torch.tensor([2.0])
    >>> m = torch.tensor([0.0])
    >>> rho = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    >>> result = zernike_polynomial_r(n, m, rho)
    >>> expected = 2 * rho**2 - 1
    >>> torch.allclose(result, expected)
    True

    Verify R_n^m(1) = 1 for valid n, m:

    >>> n = torch.tensor([0.0, 2.0, 4.0, 4.0])
    >>> m = torch.tensor([0.0, 0.0, 0.0, 2.0])
    >>> rho = torch.tensor([1.0])
    >>> zernike_polynomial_r(n, m, rho)
    tensor([1., 1., 1., 1.])

    Invalid combinations return 0:

    >>> n = torch.tensor([2.0])
    >>> m = torch.tensor([3.0])  # m > n is invalid
    >>> rho = torch.tensor([0.5])
    >>> zernike_polynomial_r(n, m, rho)
    tensor([0.])

    >>> n = torch.tensor([3.0])
    >>> m = torch.tensor([0.0])  # n-m=3 is odd, invalid
    >>> rho = torch.tensor([0.5])
    >>> zernike_polynomial_r(n, m, rho)
    tensor([0.])

    .. warning:: Gradients use finite differences

       The gradients with respect to n and m, and all second-order gradients,
       are computed using finite differences and may have reduced accuracy
       compared to analytical gradients. For integer n and m, the gradient
       is effectively zero since small perturbations don't change the result.

    Notes
    -----
    - The implementation uses the Jacobi polynomial representation for
      numerical stability.
    - The full Zernike polynomial :math:`Z_n^m(\rho, \theta)` combines the
      radial polynomial with angular terms:
      :math:`Z_n^m(\rho, \theta) = R_n^{|m|}(\rho) \cos(m\theta)` for
      :math:`m \geq 0` and
      :math:`Z_n^m(\rho, \theta) = R_n^{|m|}(\rho) \sin(|m|\theta)` for
      :math:`m < 0`.
    - The radial polynomial is symmetric in m: :math:`R_n^m = R_n^{-m}`.

    See Also
    --------
    jacobi_polynomial_p : Jacobi polynomial (used internally)
    legendre_polynomial_p : Legendre polynomial of the first kind
    """
    return torch.ops.torchscience.zernike_polynomial_r(n, m, rho)
