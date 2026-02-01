import torch
from torch import Tensor


def zernike_polynomial_z(
    n: Tensor, m: Tensor, rho: Tensor, theta: Tensor
) -> Tensor:
    r"""
    Full Zernike polynomial.

    Computes the full Zernike polynomial :math:`Z_n^m(\rho, \theta)`.

    Mathematical Definition
    -----------------------
    The full Zernike polynomial combines the radial Zernike polynomial
    :math:`R_n^{|m|}(\rho)` with angular functions:

    .. math::

       Z_n^m(\rho, \theta) = R_n^{|m|}(\rho) \cos(m\theta) \quad \text{if } m \geq 0

       Z_n^m(\rho, \theta) = R_n^{|m|}(\rho) \sin(|m|\theta) \quad \text{if } m < 0

    where :math:`R_n^m(\rho)` is the radial Zernike polynomial.

    Special Values
    --------------
    - :math:`Z_n^m(0, \theta) = 0` for :math:`|m| > 0`
    - :math:`Z_n^0(0, \theta) = R_n^0(0) = (-1)^{n/2}` for :math:`n` even
    - :math:`Z_n^m(1, \theta) = \cos(m\theta)` for :math:`m \geq 0`,
      :math:`\sin(|m|\theta)` for :math:`m < 0`

    Constraints
    -----------
    - :math:`n \geq |m| \geq 0`
    - :math:`(n - |m|)` must be even

    For invalid (n, m) combinations (:math:`n < |m|` or :math:`(n - |m|)` odd),
    returns 0.

    Standard Zernike Modes (OSA/ANSI Indexing)
    ------------------------------------------
    Note: Our implementation does not include normalization factors.

    - :math:`Z_0^0 = 1` (piston)
    - :math:`Z_1^{-1} = \rho \sin(\theta)` (y-tilt)
    - :math:`Z_1^1 = \rho \cos(\theta)` (x-tilt)
    - :math:`Z_2^{-2} = \rho^2 \sin(2\theta)` (astigmatism 45 degrees)
    - :math:`Z_2^0 = 2\rho^2 - 1` (defocus)
    - :math:`Z_2^2 = \rho^2 \cos(2\theta)` (astigmatism 0 degrees)
    - :math:`Z_3^{-1} = (3\rho^3 - 2\rho) \sin(\theta)` (y-coma)
    - :math:`Z_3^1 = (3\rho^3 - 2\rho) \cos(\theta)` (x-coma)

    Applications
    ------------
    - **Optics**: Wavefront analysis and aberration characterization in
      optical systems. Zernike polynomials form an orthonormal basis on the
      unit disk, making them ideal for representing wavefront aberrations.

    - **Adaptive optics**: Used in telescopes and microscopes to correct
      for atmospheric distortion and optical aberrations in real-time.

    - **Ophthalmology**: Standard representation for corneal topography
      and wavefront aberration measurement of the human eye.

    - **Image processing**: Zernike moments are used for rotation-invariant
      pattern recognition and image analysis.

    Parameters
    ----------
    n : Tensor
        Radial order (degree) of the polynomial. Must be a non-negative
        integer. Broadcasting with m, rho, and theta is supported.
    m : Tensor
        Azimuthal frequency (angular order). Must satisfy :math:`|m| \leq n`
        and :math:`(n - |m|)` must be even. Negative m values use the sine
        angular term, positive m values use the cosine term.
        Broadcasting with n, rho, and theta is supported.
    rho : Tensor
        Radial coordinate, typically in the range [0, 1] for the unit disk.
        Broadcasting with n, m, and theta is supported.
    theta : Tensor
        Angular coordinate in radians. Broadcasting with n, m, and rho
        is supported.

    Returns
    -------
    Tensor
        The full Zernike polynomial :math:`Z_n^m(\rho, \theta)` evaluated
        at the input values.

    Examples
    --------
    Piston mode Z_0^0:

    >>> n = torch.tensor([0.0])
    >>> m = torch.tensor([0.0])
    >>> rho = torch.tensor([0.5])
    >>> theta = torch.tensor([0.0])
    >>> zernike_polynomial_z(n, m, rho, theta)
    tensor([1.])

    X-tilt mode Z_1^1 = rho * cos(theta):

    >>> import math
    >>> n = torch.tensor([1.0])
    >>> m = torch.tensor([1.0])
    >>> rho = torch.tensor([1.0])
    >>> theta = torch.tensor([0.0])  # cos(0) = 1
    >>> zernike_polynomial_z(n, m, rho, theta)
    tensor([1.])

    Y-tilt mode Z_1^{-1} = rho * sin(theta):

    >>> n = torch.tensor([1.0])
    >>> m = torch.tensor([-1.0])
    >>> rho = torch.tensor([1.0])
    >>> theta = torch.tensor([math.pi / 2])  # sin(pi/2) = 1
    >>> result = zernike_polynomial_z(n, m, rho, theta)
    >>> torch.allclose(result, torch.tensor([1.0]))
    True

    Defocus mode Z_2^0:

    >>> n = torch.tensor([2.0])
    >>> m = torch.tensor([0.0])
    >>> rho = torch.tensor([0.5], dtype=torch.float64)
    >>> theta = torch.tensor([0.0], dtype=torch.float64)
    >>> result = zernike_polynomial_z(n, m, rho, theta)
    >>> expected = 2 * 0.5**2 - 1  # = -0.5
    >>> torch.allclose(result, torch.tensor([expected]))
    True

    .. warning:: Gradients use finite differences

       The gradients with respect to n and m, and all second-order gradients,
       are computed using finite differences and may have reduced accuracy
       compared to analytical gradients. For integer n and m, the gradient
       is effectively zero since small perturbations don't change the result.

    Notes
    -----
    - The Zernike polynomials form a complete orthonormal basis over the
      unit disk when normalized. The normalization factors are:

      .. math::

         N_n^m = \sqrt{\frac{2(n+1)}{1 + \delta_{m0}}}

      where :math:`\delta_{m0}` is the Kronecker delta. Our implementation
      does not include these normalization factors.

    - The polynomials are orthogonal over the unit disk:

      .. math::

         \int_0^{2\pi} \int_0^1 Z_n^m Z_{n'}^{m'} \rho \, d\rho \, d\theta
         = \frac{\pi}{N_n^m} \delta_{nn'} \delta_{mm'}

    See Also
    --------
    zernike_polynomial_r : Radial Zernike polynomial (used internally)
    jacobi_polynomial_p : Jacobi polynomial (underlying mathematical function)
    """
    return torch.ops.torchscience.zernike_polynomial_z(n, m, rho, theta)
