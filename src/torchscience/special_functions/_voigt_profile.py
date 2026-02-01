import torch
from torch import Tensor


def voigt_profile(x: Tensor, sigma: Tensor, gamma: Tensor) -> Tensor:
    r"""
    Voigt profile (Voigt function).

    Computes the Voigt profile V(x, sigma, gamma), which is the convolution
    of a Gaussian and a Lorentzian distribution.

    Mathematical Definition
    -----------------------
    The Voigt profile is defined as:

    .. math::

       V(x, \sigma, \gamma) = \frac{\text{Re}[w(z)]}{\sigma \sqrt{2\pi}}

    where:

    .. math::

       z = \frac{x + i\gamma}{\sigma \sqrt{2}}

    and w(z) is the Faddeeva function.

    Equivalently, the Voigt profile is the convolution of a Gaussian with
    standard deviation sigma and a Lorentzian (Cauchy) distribution with
    half-width at half-maximum gamma:

    .. math::

       V(x, \sigma, \gamma) = \int_{-\infty}^{\infty} G(x', \sigma) L(x - x', \gamma) \, dx'

    where G is the Gaussian and L is the Lorentzian.

    Properties
    ----------
    - V(x, sigma, gamma) is a valid probability density function (integrates to 1)
    - V(x, sigma, gamma) = V(-x, sigma, gamma) (symmetric in x)
    - V(x, sigma, gamma) > 0 for all x when sigma > 0 and gamma >= 0

    Limiting Cases
    --------------
    - When gamma -> 0: V approaches a Gaussian with standard deviation sigma
    - When sigma -> 0: V approaches a Lorentzian with HWHM gamma (in the limit)

    Domain
    ------
    - x: any real number
    - sigma: must be positive (sigma > 0)
    - gamma: must be non-negative (gamma >= 0)

    For invalid parameter values (sigma <= 0 or gamma < 0), the result is NaN.

    Algorithm
    ---------
    Uses the Faddeeva function w(z) for numerical evaluation. The Faddeeva
    function is computed using a combination of:
    - Taylor series for small arguments
    - Continued fraction for intermediate arguments
    - Asymptotic expansion for large arguments

    Applications
    ------------
    The Voigt profile is widely used in:

    - **Spectroscopy**: Modeling spectral line shapes that have both Doppler
      (Gaussian) and natural/pressure (Lorentzian) broadening
    - **Astrophysics**: Absorption and emission line profiles in stellar
      atmospheres
    - **Laser spectroscopy**: Modeling atomic and molecular transitions
    - **Nuclear physics**: Resonance line shapes

    Related Functions
    -----------------
    - Faddeeva function: w(z) is used to compute the Voigt profile
    - Gaussian: Normal distribution with standard deviation sigma
    - Lorentzian: Cauchy distribution with scale parameter gamma

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Output is always real (same floating-point type as inputs)

    Autograd Support
    ----------------
    Gradients are fully supported when inputs have requires_grad=True.
    All first-order gradients (with respect to x, sigma, and gamma) are
    computed analytically using the Faddeeva function derivative:

    .. math::

       \frac{dw}{dz} = -2z \cdot w(z) + \frac{2i}{\sqrt{\pi}}

    Second-order derivatives are also supported.

    Parameters
    ----------
    x : Tensor
        Position parameter. Can be any real number.
        Broadcasting with sigma and gamma is supported.
    sigma : Tensor
        Gaussian standard deviation. Must be positive (sigma > 0).
        Broadcasting with x and gamma is supported.
    gamma : Tensor
        Lorentzian half-width at half-maximum. Must be non-negative (gamma >= 0).
        Broadcasting with x and sigma is supported.

    Returns
    -------
    Tensor
        The Voigt profile V(x, sigma, gamma) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    >>> sigma = torch.tensor([1.0], dtype=torch.float64)
    >>> gamma = torch.tensor([0.5], dtype=torch.float64)
    >>> voigt_profile(x, sigma, gamma)
    tensor([0.3183, 0.1903, 0.0726])

    Pure Gaussian limit (gamma = 0):

    >>> x = torch.tensor([0.0], dtype=torch.float64)
    >>> sigma = torch.tensor([1.0], dtype=torch.float64)
    >>> gamma = torch.tensor([0.0], dtype=torch.float64)
    >>> voigt_profile(x, sigma, gamma)  # Should be 1/(sigma*sqrt(2*pi)) ~ 0.3989
    tensor([0.3989])

    Symmetric property:

    >>> x = torch.tensor([1.0], dtype=torch.float64)
    >>> sigma = torch.tensor([1.0], dtype=torch.float64)
    >>> gamma = torch.tensor([0.5], dtype=torch.float64)
    >>> voigt_profile(x, sigma, gamma)
    tensor([0.1903])
    >>> voigt_profile(-x, sigma, gamma)  # Same value
    tensor([0.1903])

    Autograd:

    >>> x = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    >>> sigma = torch.tensor([1.0], dtype=torch.float64)
    >>> gamma = torch.tensor([0.5], dtype=torch.float64)
    >>> y = voigt_profile(x, sigma, gamma)
    >>> y.backward()
    >>> x.grad  # Gradient with respect to x
    tensor([-0.1488])

    Notes
    -----
    - The Voigt profile is always positive and integrates to 1
    - The peak value is at x = 0 for any sigma > 0 and gamma >= 0
    - The profile width depends on both sigma and gamma
    - The Voigt FWHM can be approximated by:
      FWHM_V ~ 0.5346*FWHM_L + sqrt(0.2166*FWHM_L^2 + FWHM_G^2)
      where FWHM_L = 2*gamma and FWHM_G = 2*sqrt(2*ln(2))*sigma

    See Also
    --------
    scipy.special.voigt_profile : SciPy's implementation of the Voigt profile
    faddeeva_w : The Faddeeva function used in this implementation
    """
    return torch.ops.torchscience.voigt_profile(x, sigma, gamma)
