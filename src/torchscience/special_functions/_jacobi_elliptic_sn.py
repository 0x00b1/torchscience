import torch
from torch import Tensor


def jacobi_elliptic_sn(u: Tensor, m: Tensor) -> Tensor:
    r"""
    Jacobi elliptic function sn(u, m).

    Computes the Jacobi elliptic sine function evaluated at each element
    of the input tensors.

    Mathematical Definition
    -----------------------
    The Jacobi elliptic function sn is defined as:

    .. math::

       \text{sn}(u, m) = \sin(\text{am}(u, m))

    where am(u, m) is the Jacobi amplitude function, defined implicitly by:

    .. math::

       u = \int_0^{\text{am}(u, m)} \frac{d\theta}{\sqrt{1 - m \sin^2\theta}}

    This function uses the parameter convention where m is the "parameter"
    (not the modulus k). The relationship is m = k^2.

    Domain
    ------
    - u: any real or complex value (the argument)
    - m: elliptic parameter (conventionally 0 <= m <= 1 for real values)
    - For m < 0 or m > 1, analytic continuation is used

    Special Values
    --------------
    - sn(0, m) = 0 for all m
    - sn(K(m), m) = 1 where K(m) is the complete elliptic integral
    - sn(u, 0) = sin(u) (circular limit)
    - sn(u, 1) = tanh(u) (hyperbolic limit)
    - sn(-u, m) = -sn(u, m) (odd function in u)

    Period Structure
    ----------------
    The sn function is doubly periodic in the complex plane:
    - sn(u + 4K(m), m) = sn(u, m) (real period 4K(m))
    - sn(u + 2iK'(m), m) = sn(u, m) (imaginary period 2iK'(m))

    where K(m) is the complete elliptic integral of the first kind and
    K'(m) = K(1-m) is the complementary complete elliptic integral.

    Relationship to Other Jacobi Functions
    --------------------------------------
    The three primary Jacobi elliptic functions satisfy:

    .. math::

       \text{sn}^2(u, m) + \text{cn}^2(u, m) = 1

       \text{dn}^2(u, m) + m \cdot \text{sn}^2(u, m) = 1

    Algorithm
    ---------
    The implementation uses the Landen transformation (descending sequence)
    to compute the Jacobi amplitude function, then returns sin(am).
    This approach provides excellent numerical accuracy across the entire
    domain.

    Applications
    ------------
    The Jacobi elliptic function sn appears in many mathematical and
    physical contexts:
    - Exact solution to the pendulum equation (arbitrary amplitude)
    - Conformal mapping of rectangles
    - Cnoidal waves in shallow water theory
    - Duffing oscillator solutions
    - Elliptic filter design in signal processing
    - Soliton solutions to nonlinear wave equations

    Dtype Promotion
    ---------------
    - All inputs are promoted to a common dtype
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when inputs require grad.
    The gradient with respect to u is:

    .. math::

       \frac{\partial \text{sn}}{\partial u} = \text{cn}(u, m) \cdot \text{dn}(u, m)

    The gradient with respect to m involves more complex expressions.
    This implementation uses numerical differentiation for both gradients.

    Second-order derivatives (gradgradcheck) are also supported via
    numerical differentiation.

    Parameters
    ----------
    u : Tensor
        The argument (elliptic argument). Can be floating-point or complex.
        Broadcasting with m is supported.
    m : Tensor
        The elliptic parameter. Conventionally 0 <= m <= 1 for real values.
        Broadcasting with u is supported.

    Returns
    -------
    Tensor
        The Jacobi elliptic function sn(u, m) evaluated at the input values.
        Output dtype follows promotion rules.

    Examples
    --------
    Basic usage:

    >>> u = torch.tensor([0.0])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_sn(u, m)  # sn(0, m) = 0
    tensor([0.])

    Circular limit (m = 0):

    >>> u = torch.tensor([1.0])
    >>> m = torch.tensor([0.0])
    >>> jacobi_elliptic_sn(u, m)  # equals sin(1.0)
    tensor([0.8415])

    Hyperbolic limit (m = 1):

    >>> u = torch.tensor([1.0])
    >>> m = torch.tensor([1.0])
    >>> jacobi_elliptic_sn(u, m)  # equals tanh(1.0)
    tensor([0.7616])

    Multiple values:

    >>> u = torch.tensor([0.0, 0.5, 1.0, 1.5])
    >>> m = torch.tensor([0.5])
    >>> jacobi_elliptic_sn(u, m)
    tensor([0.0000, 0.4750, 0.8031, 0.9689])

    Complex input:

    >>> u = torch.tensor([1.0 + 0.5j])
    >>> m = torch.tensor([0.5 + 0.0j])
    >>> jacobi_elliptic_sn(u, m)
    tensor([0.8392+0.2512j])

    Autograd:

    >>> u = torch.tensor([1.0], requires_grad=True)
    >>> m = torch.tensor([0.5])
    >>> sn = jacobi_elliptic_sn(u, m)
    >>> sn.backward()
    >>> u.grad  # d(sn)/du = cn(u,m) * dn(u,m)
    tensor([0.5234])

    Notes
    -----
    - This function uses the parameter convention (m), not the modulus
      convention (k). If you have the modulus k, use m = k^2.
    - The implementation uses the Landen transformation for numerical
      stability and accuracy.
    - For m very close to 0, the circular approximation sin(u) is used.
    - For m very close to 1, the hyperbolic approximation tanh(u) is used.

    See Also
    --------
    jacobi_elliptic_cn : Jacobi elliptic function cn(u, m)
    jacobi_elliptic_dn : Jacobi elliptic function dn(u, m)
    jacobi_amplitude_am : Jacobi amplitude function am(u, m)
    complete_legendre_elliptic_integral_k : Complete elliptic integral K(m)
    """
    return torch.ops.torchscience.jacobi_elliptic_sn(u, m)
