import torch
from torch import Tensor


def faddeeva_w(z: Tensor) -> Tensor:
    r"""
    Faddeeva function (scaled complex complementary error function).

    Computes the Faddeeva function w(z) evaluated at each element of the
    input tensor.

    Mathematical Definition
    -----------------------
    The Faddeeva function is defined as:

    .. math::

       w(z) = e^{-z^2} \mathrm{erfc}(-iz)

    where erfc is the complementary error function and i is the imaginary unit.

    Equivalently, the Faddeeva function can be expressed as:

    .. math::

       w(z) = \frac{i}{\pi} \int_{-\infty}^{\infty} \frac{e^{-t^2}}{z - t} dt

    which is the Hilbert transform of the Gaussian function divided by sqrt(pi).

    Special Values
    --------------
    - w(0) = 1
    - w(x) for real x > 0: real part is exp(-x^2), imaginary part is 2*dawson(x)
    - w(iy) for imaginary y > 0: w(iy) = erfc(y) (purely real)
    - Asymptotic: w(z) ~ i/(sqrt(pi)*z) for large |z| with Im(z) > 0

    Symmetry Properties
    -------------------
    - Reflection: w(-z) = 2*exp(-z^2) - w(z)
    - Complex conjugate: w(conj(z)) = conj(w(-z)) for Im(z) > 0

    Domain
    ------
    - z: any complex number
    - The function is entire (analytic everywhere in the complex plane)
    - No poles or branch cuts

    Algorithm
    ---------
    - Uses a region-based approach following Algorithm 916 (Zaghloul & Ali, 2012):
      - Small |z|: Taylor series expansion
      - Intermediate |z|: Continued fraction (Laplace representation)
      - Large |z|: Asymptotic expansion
    - For z in lower half-plane (Im(z) < 0), uses the reflection formula

    Applications
    ------------
    The Faddeeva function is fundamental in several areas:

    - **Plasma physics**: The plasma dispersion function Z(z) = i*sqrt(pi)*w(z)
      appears in kinetic theory of plasmas and wave propagation
    - **Spectroscopy**: The Voigt profile (convolution of Gaussian and Lorentzian)
      is proportional to Re[w(z)] for appropriate z
    - **Signal processing**: Related to the Hilbert transform of Gaussian functions
    - **Error functions**: Foundation for computing complex error functions,
      Dawson function, and other related special functions

    Related Functions
    -----------------
    - Dawson function: D(x) = sqrt(pi)/2 * Im[w(x)] for real x
    - Voigt profile: V(x, sigma, gamma) ~ Re[w((x + i*gamma)/(sqrt(2)*sigma))]
    - Complex error function: erf(z) = 1 - exp(-z^2)*w(iz)

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Real inputs produce complex outputs
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dz} w(z) = -2z \cdot w(z) + \frac{2i}{\sqrt{\pi}}

    Second-order derivatives (gradgradcheck) are also supported, computed
    using:

    .. math::

       \frac{d^2}{dz^2} w(z) = (4z^2 - 2) w(z) - \frac{4iz}{\sqrt{\pi}}

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Faddeeva function w(z) evaluated at each element of z.
        Output dtype is always complex (complex64 or complex128).

    Examples
    --------
    Basic evaluation at real argument:

    >>> z = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    >>> faddeeva_w(z)
    tensor([1.0000+0.0000j, 0.3679+0.6072j, 0.0183+0.3321j])

    At the origin:

    >>> z = torch.tensor([0.0 + 0.0j], dtype=torch.complex128)
    >>> faddeeva_w(z)
    tensor([1.+0.j])

    On the imaginary axis (gives real erfc values):

    >>> y = torch.tensor([1.0], dtype=torch.float64)
    >>> z = torch.tensor([1.0j], dtype=torch.complex128)
    >>> faddeeva_w(z)  # Should be approximately erfc(1) = 0.1573
    tensor([0.1573+0.j])

    Autograd:

    >>> z = torch.tensor([1.0 + 0.5j], dtype=torch.complex128, requires_grad=True)
    >>> y = faddeeva_w(z)
    >>> y.abs().backward()
    >>> z.grad is not None
    True

    Notes
    -----
    - The function is numerically stable across the entire complex plane
    - For applications involving only real arguments, the Dawson function
      may be more efficient to compute directly
    - The plasma dispersion function Z(z) = i*sqrt(pi)*w(z) is available
      as a separate function in some libraries

    See Also
    --------
    scipy.special.wofz : SciPy's implementation of the Faddeeva function
    """
    return torch.ops.torchscience.faddeeva_w(z)
