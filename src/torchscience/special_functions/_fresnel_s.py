import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def fresnel_s(z: Tensor) -> Tensor:
    r"""
    Fresnel sine integral S(z).

    Computes the Fresnel sine integral evaluated at each element of the input
    tensor.

    Mathematical Definition
    -----------------------
    The Fresnel sine integral is defined as:

    .. math::

       S(z) = \int_{0}^{z} \sin\left(\frac{\pi t^2}{2}\right) \, dt

    S(z) is an odd entire function, meaning it has no singularities in the
    finite complex plane and satisfies S(-z) = -S(z).

    Series Expansion
    ----------------
    S(z) has a convergent Taylor series for all z:

    .. math::

       S(z) = \sum_{n=0}^{\infty} \frac{(-1)^n \left(\frac{\pi}{2}\right)^{2n+1}
              z^{4n+3}}{(2n+1)! \cdot (4n+3)}

    Asymptotic Behavior
    -------------------
    For large |z|:

    .. math::

       S(z) = \frac{1}{2} - f(z)\cos\left(\frac{\pi z^2}{2}\right)
              - g(z)\sin\left(\frac{\pi z^2}{2}\right) + O(z^{-5})

    where f(z) and g(z) are auxiliary functions that decay as 1/(pi*z) and
    1/(pi*z)^2 respectively.

    Special Values
    --------------
    - S(0) = 0
    - S(+inf) = 0.5
    - S(-inf) = -0.5
    - S is an odd function: S(-z) = -S(z)

    Domain
    ------
    - z: any real or complex value
    - S(z) is entire, so there are no singularities

    Derivatives
    -----------
    The derivative of S(z) is the integrand:

    .. math::

       \frac{d}{dz} S(z) = \sin\left(\frac{\pi z^2}{2}\right)

    The second derivative is:

    .. math::

       \frac{d^2}{dz^2} S(z) = \pi z \cos\left(\frac{\pi z^2}{2}\right)

    Applications
    ------------
    The Fresnel sine integral appears in:

    - Optics and diffraction theory (Fresnel diffraction, Cornu spiral)
    - Civil engineering (road and railway design, clothoid curves)
    - Signal processing
    - Electromagnetic theory
    - Antenna design

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs must be explicitly converted to floating-point types

    Autograd Support
    ----------------
    Full autograd support including second-order derivatives.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The Fresnel sine integral S evaluated at each element of z. Output
        dtype matches input dtype.

    Examples
    --------
    Basic evaluation:

    >>> z = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
    >>> fresnel_s(z)
    tensor([0.0000, 0.0647, 0.4383, 0.3434, 0.4992])

    Asymptotic behavior (approaches 0.5):

    >>> z = torch.tensor([10.0, 20.0, 50.0])
    >>> fresnel_s(z)
    tensor([0.4682, 0.4669, 0.4936])

    Odd function property:

    >>> z = torch.tensor([1.0, -1.0])
    >>> fresnel_s(z)
    tensor([ 0.4383, -0.4383])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j], dtype=torch.complex64)
    >>> fresnel_s(z)
    tensor([0.4527+0.2031j])

    Autograd:

    >>> z = torch.tensor([1.0], requires_grad=True)
    >>> y = fresnel_s(z)
    >>> y.backward()
    >>> z.grad  # sin(pi/2) = 1.0
    tensor([1.0000])

    The derivative at z=0:

    >>> z = torch.tensor([0.0], requires_grad=True)
    >>> y = fresnel_s(z)
    >>> y.backward()
    >>> z.grad  # sin(0) = 0.0
    tensor([0.])

    See Also
    --------
    fresnel_c : Fresnel cosine integral C(z)
    """
    return torch.ops.torchscience.fresnel_s(z)
