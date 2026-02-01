import torch
from torch import Tensor

import torchscience._csrc  # noqa: F401 - registers operators


def fresnel_c(z: Tensor) -> Tensor:
    r"""
    Fresnel cosine integral C(z).

    Computes the Fresnel cosine integral evaluated at each element of the input
    tensor.

    Mathematical Definition
    -----------------------
    The Fresnel cosine integral is defined as:

    .. math::

       C(z) = \int_{0}^{z} \cos\left(\frac{\pi t^2}{2}\right) \, dt

    C(z) is an odd entire function, meaning it has no singularities in the
    finite complex plane and satisfies C(-z) = -C(z).

    Series Expansion
    ----------------
    C(z) has a convergent Taylor series for all z:

    .. math::

       C(z) = \sum_{n=0}^{\infty} \frac{(-1)^n \left(\frac{\pi}{2}\right)^{2n}
              z^{4n+1}}{(2n)! \cdot (4n+1)}

    Asymptotic Behavior
    -------------------
    For large |z|:

    .. math::

       C(z) = \frac{1}{2} + f(z)\sin\left(\frac{\pi z^2}{2}\right)
              - g(z)\cos\left(\frac{\pi z^2}{2}\right) + O(z^{-5})

    where f(z) and g(z) are auxiliary functions that decay as 1/(pi*z) and
    1/(pi*z)^2 respectively.

    Special Values
    --------------
    - C(0) = 0
    - C(+inf) = 0.5
    - C(-inf) = -0.5
    - C is an odd function: C(-z) = -C(z)

    Domain
    ------
    - z: any real or complex value
    - C(z) is entire, so there are no singularities

    Derivatives
    -----------
    The derivative of C(z) is the integrand:

    .. math::

       \frac{d}{dz} C(z) = \cos\left(\frac{\pi z^2}{2}\right)

    The second derivative is:

    .. math::

       \frac{d^2}{dz^2} C(z) = -\pi z \sin\left(\frac{\pi z^2}{2}\right)

    Applications
    ------------
    The Fresnel cosine integral appears in:

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
        The Fresnel cosine integral C evaluated at each element of z. Output
        dtype matches input dtype.

    Examples
    --------
    Basic evaluation:

    >>> z = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
    >>> fresnel_c(z)
    tensor([0.0000, 0.4923, 0.7799, 0.4883, 0.5636])

    Asymptotic behavior (approaches 0.5):

    >>> z = torch.tensor([10.0, 20.0, 50.0])
    >>> fresnel_c(z)
    tensor([0.4999, 0.5099, 0.4999])

    Odd function property:

    >>> z = torch.tensor([1.0, -1.0])
    >>> fresnel_c(z)
    tensor([ 0.7799, -0.7799])

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j], dtype=torch.complex64)
    >>> fresnel_c(z)
    tensor([0.8109-0.0946j])

    Autograd:

    >>> z = torch.tensor([1.0], requires_grad=True)
    >>> y = fresnel_c(z)
    >>> y.backward()
    >>> z.grad  # cos(pi/2) = 0.0
    tensor([0.])

    The derivative at z=0:

    >>> z = torch.tensor([0.0], requires_grad=True)
    >>> y = fresnel_c(z)
    >>> y.backward()
    >>> z.grad  # cos(0) = 1.0
    tensor([1.])

    See Also
    --------
    fresnel_s : Fresnel sine integral S(z)
    """
    return torch.ops.torchscience.fresnel_c(z)
