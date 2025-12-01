import torch
from torch import Tensor


def complete_legendre_elliptic_integral_k(m: Tensor) -> Tensor:
    r"""
    Complete elliptic integral of the first kind K(m).

    Computes the complete Legendre elliptic integral of the first kind,
    evaluated at each element of the input tensor.

    Mathematical Definition
    -----------------------
    The complete elliptic integral of the first kind is defined as:

    .. math::

       K(m) = \int_0^{\pi/2} \frac{d\theta}{\sqrt{1 - m \sin^2\theta}}

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Relation to Carlson Symmetric Form
    ----------------------------------
    K(m) can be expressed using Carlson's symmetric elliptic integral R_F:

    .. math::

       K(m) = R_F(0, 1-m, 1)

    Domain
    ------
    - For real m: m < 1 (singularity at m = 1 where K diverges logarithmically)
    - For complex m: entire complex plane with branch cut at [1, infinity)
    - K(0) = pi/2

    Special Values
    --------------
    - K(0) = pi/2
    - K(0.5) approximately equals 1.8541
    - K(m) -> infinity as m -> 1 (logarithmic singularity)

    Applications
    ------------
    The complete elliptic integral of the first kind appears in many
    mathematical and physical contexts:
    - Period of a simple pendulum for arbitrary amplitudes
    - Circumference of an ellipse
    - Conformal mapping applications
    - Solutions to the Laplace equation in elliptic coordinates
    - Electromagnetic field calculations
    - Geodesics on ellipsoids

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when m.requires_grad is True.
    The gradient is computed using the formula:

    .. math::

       \frac{dK}{dm} = \frac{E(m) - (1-m)K(m)}{2m(1-m)}

    where E(m) is the complete elliptic integral of the second kind.

    Second-order derivatives (gradgradcheck) are also supported.

    Parameters
    ----------
    m : Tensor
        Input tensor (the parameter). Can be floating-point or complex.

    Returns
    -------
    Tensor
        The complete elliptic integral of the first kind K(m) evaluated
        at each element of m. Output dtype matches input dtype (or
        promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> m = torch.tensor([0.0])
    >>> complete_legendre_elliptic_integral_k(m)  # pi/2
    tensor([1.5708])

    Standard test value:

    >>> m = torch.tensor([0.5])
    >>> complete_legendre_elliptic_integral_k(m)  # approximately 1.8541
    tensor([1.8541])

    Multiple values:

    >>> m = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9])
    >>> complete_legendre_elliptic_integral_k(m)
    tensor([1.5708, 1.6858, 1.8541, 2.1565, 2.5781])

    Complex input:

    >>> m = torch.tensor([0.5 + 0.1j])
    >>> complete_legendre_elliptic_integral_k(m)
    tensor([1.8449-0.0681j])

    Autograd:

    >>> m = torch.tensor([0.5], requires_grad=True)
    >>> K = complete_legendre_elliptic_integral_k(m)
    >>> K.backward()
    >>> m.grad
    tensor([0.8472])

    .. warning:: Singularity at m = 1

       The function has a logarithmic singularity at m = 1:

       .. math::

          K(m) \sim \ln\frac{4}{\sqrt{1-m}} \quad \text{as } m \to 1^-

       The function returns infinity at m = 1.

    .. warning:: Branch cut for m > 1

       For real m > 1, the function involves complex values. Use complex
       input tensors for analytic continuation beyond m = 1.

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The implementation uses Carlson's R_F integral, which provides
      excellent numerical stability and accuracy.
    - For the complementary modulus m' = 1 - m, the relationship is:
      K(m) = R_F(0, m', 1)

    See Also
    --------
    complete_legendre_elliptic_integral_e : Complete elliptic integral of the second kind
    complete_legendre_elliptic_integral_pi : Complete elliptic integral of the third kind
    carlson_elliptic_integral_r_f : Carlson's symmetric elliptic integral R_F
    """
    return torch.ops.torchscience.complete_legendre_elliptic_integral_k(m)
