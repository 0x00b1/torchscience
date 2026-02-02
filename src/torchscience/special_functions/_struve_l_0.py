import torch
from torch import Tensor


def struve_l_0(z: Tensor) -> Tensor:
    r"""
    Modified Struve function of order zero.

    Computes the modified Struve function L_0(z) evaluated at each element
    of the input tensor.

    Mathematical Definition
    -----------------------
    The modified Struve function of order zero is defined as:

    .. math::

       \mathbf{L}_0(z) = \sum_{k=0}^\infty \frac{(z/2)^{2k+1}}{(\Gamma(k + 3/2))^2}

    This is related to the regular Struve function H_0 by:

    .. math::

       \mathbf{L}_0(z) = -i \mathbf{H}_0(iz)

    Special Values
    --------------
    - L_0(0) = 0
    - L_0(+inf) = +inf (grows exponentially like I_0(z))
    - L_0(-inf) = -inf (odd function)
    - L_0(NaN) = NaN

    Symmetry
    --------
    L_0 is an odd function: L_0(-z) = -L_0(z)

    Domain
    ------
    - z: any real or complex value
    - L_0 is an entire function (no singularities or branch cuts)
    - For complex z, accuracy is best near the real axis

    Algorithm
    ---------
    - Uses power series expansion for small |z|
    - The power series has the same form as H_0 but without alternating signs
    - Half-precision types (float16, bfloat16) compute in float32 for accuracy

    Relation to Modified Bessel Function
    -------------------------------------
    For large positive z:

    .. math::

       \mathbf{L}_0(z) \sim I_0(z) - \frac{2}{\pi}

    where I_0(z) is the modified Bessel function of the first kind.

    Applications
    ------------
    The modified Struve function L_0 appears in many contexts:
    - Electromagnetics: electromagnetic field calculations
    - Heat conduction: certain heat flow problems in cylindrical geometry
    - Potential theory: Green's functions for certain boundary problems
    - Plasma physics: electron kinetic equations

    Dtype Promotion
    ---------------
    - Supports float16, bfloat16, float32, float64
    - Supports complex64, complex128
    - Integer inputs are promoted to floating-point types

    Autograd Support
    ----------------
    Gradients are fully supported when z.requires_grad is True.
    The gradient is computed using:

    .. math::

       \frac{d}{dz} \mathbf{L}_0(z) = \frac{2}{\pi} + \mathbf{L}_1(z)

    Note the positive sign, unlike H_0 which has d/dz H_0(z) = 2/pi - H_1(z).

    Second-order derivatives are also supported.

    Parameters
    ----------
    z : Tensor
        Input tensor. Can be floating-point or complex.

    Returns
    -------
    Tensor
        The modified Struve function L_0 evaluated at each element of z.
        Output dtype matches input dtype (or promoted dtype for integers).

    Examples
    --------
    Basic usage:

    >>> z = torch.tensor([0.0, 1.0, 2.0, 3.0])
    >>> struve_l_0(z)
    tensor([0.0000, 0.6727, 2.0257, 4.7472])

    Odd function property:

    >>> z = torch.tensor([1.0, 2.0, 3.0])
    >>> torch.allclose(struve_l_0(-z), -struve_l_0(z))
    True

    Complex input:

    >>> z = torch.tensor([1.0 + 0.5j, 2.0 - 0.5j])
    >>> struve_l_0(z)
    tensor([...])  # Complex output

    Autograd:

    >>> z = torch.tensor([2.0], requires_grad=True)
    >>> y = struve_l_0(z)
    >>> y.backward()
    >>> z.grad  # equals 2/pi + L_1(2.0)
    tensor([...])

    Asymptotic behavior (approaches I_0(z) - 2/pi for large z):

    >>> import scipy.special
    >>> z_val = 10.0
    >>> l0 = struve_l_0(torch.tensor([z_val])).item()
    >>> i0 = scipy.special.i0(z_val)
    >>> two_over_pi = 2.0 / 3.14159265
    >>> # l0 is approximately i0 - two_over_pi for large z

    Notes
    -----
    - The modified Struve function is related to the regular Struve function
      by L_n(z) = -i^{-n-1} H_n(iz) for integer n.
    - L_0(z) grows exponentially for large positive z, unlike H_0 which oscillates.
    - Complex accuracy: approximations are optimized for real
      arguments. For complex z with |Im(z)| > |Re(z)|, accuracy may degrade.

    See Also
    --------
    struve_l_1 : Modified Struve function of order one
    struve_h_0 : Struve function of order zero
    modified_bessel_i_0 : Modified Bessel function of the first kind of order zero
    """
    return torch.ops.torchscience.struve_l_0(z)
